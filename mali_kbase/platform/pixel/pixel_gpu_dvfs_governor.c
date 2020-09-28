// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright 2020 Google LLC.
 *
 * Author: Sidath Senanayake <sidaths@google.com>
 */

/* Mali core includes */
#include <mali_kbase.h>

/* Pixel integration includes */
#include "mali_kbase_config_platform.h"
#include "pixel_gpu_control.h"
#include "pixel_gpu_debug.h"
#include "pixel_gpu_dvfs.h"

/**
 * gpu_dvfs_governor_basic() - The evaluation function for &GPU_DVFS_GOVERNOR_BASIC.
 *
 * @kbdev: The &struct kbase_device for the GPU.
 * @util:  The current GPU utilization percentage.
 *
 * Return: The level that the GPU should run at next.
 *
 * Context: Process context. Expects the caller to hold the DVFS lock.
 */
static int gpu_dvfs_governor_basic(struct kbase_device *kbdev, int util)
{
	struct pixel_context *pc = kbdev->platform_context;
	struct gpu_dvfs_opp *tbl = pc->dvfs.table;
	int level = pc->dvfs.level;
	int level_max = pc->dvfs.level_max;
	int level_min = pc->dvfs.level_min;

	lockdep_assert_held(&pc->dvfs.lock);

	if ((level > level_max) && (util > tbl[level].util_max)) {
		/* Need to clock up*/
		level--;

		/* Reset hysteresis */
		pc->dvfs.governor.delay = tbl[level].hysteresis;

	} else if ((level < level_min) && (util < tbl[level].util_min)) {
		/* We are clocked too high */
		pc->dvfs.governor.delay--;

		/* Check if we've resisted downclocking long enough */
		if (pc->dvfs.governor.delay == 0) {
			/* Time to clock down */
			level++;

			/* Reset hysteresis */
			pc->dvfs.governor.delay = tbl[level].hysteresis;
		}
	} else {
		/* We are at the correct level, reset hysteresis */
		pc->dvfs.governor.delay = tbl[level].hysteresis;
	}

	return clamp(level, level_max, level_min);
}

static struct gpu_dvfs_governor_info governors[GPU_DVFS_GOVERNOR_COUNT] = {
	{
		GPU_DVFS_GOVERNOR_BASIC,
		"basic",
		gpu_dvfs_governor_basic,
	}
};

/**
 * gpu_dvfs_governor_get_next_level() - Requests the current governor to suggest the next level.
 *
 * @kbdev: The &struct kbase_device for the GPU.
 * util:   The utilization percentage on the GPU.
 *
 * This function ensures that the recommended level conforms to any extant
 * clock limits.
 *
 * Return: Returns the level the GPU should run at.
 *
 * Context: Process context. Expects the caller to hold the DVFS lock.
 */
int gpu_dvfs_governor_get_next_level(struct kbase_device *kbdev, int util)
{
	struct pixel_context *pc = kbdev->platform_context;
	int level, level_min, level_max;

	lockdep_assert_held(&pc->dvfs.lock);

	level_min = pc->dvfs.level_scaling_min;
	level_max = pc->dvfs.level_scaling_max;

#ifdef CONFIG_MALI_PIXEL_GPU_THERMAL
	/*
	 * If we have a TMU limit enforced, we restrict what the recommended
	 * level will be. However, we do allow overriding the TMU limit by
	 * setting scaling_min_level. Therefore thre is no adjustment to
	 * level_min below.
	 */
	level_max = max(level_max, pc->dvfs.tmu.level_limit);
#endif /* CONFIG_MALI_PIXEL_GPU_THERMAL */

	level = governors[pc->dvfs.governor.curr].evaluate(kbdev, util);

	return clamp(level, level_max, level_min);
}

/**
 * gpu_dvfs_governor_set_governor() - Sets the currently active DVFS governor.
 *
 * @kbdev: The &struct kbase_device for the GPU.
 * @gov:   &enum gpu_dvfs_governor value of the governor to set.
 *
 * Return: On success returns 0. If @gov is invalid, -EINVAL is returned.
 *
 * Context: Expects the caller to hold the DVFS lock.
 */
int gpu_dvfs_governor_set_governor(struct kbase_device *kbdev, enum gpu_dvfs_governor_type gov)
{
	struct pixel_context *pc = kbdev->platform_context;

	lockdep_assert_held(&pc->dvfs.lock);

	if (gov < 0 || gov >= GPU_DVFS_GOVERNOR_COUNT) {
		GPU_LOG(LOG_WARN, kbdev, "Attempted to set invalid DVFS governor\n");
		return -EINVAL;
	}

	pc->dvfs.governor.curr = gov;

	return 0;
}

/**
 * gpu_dvfs_governor_get_id() - Given a valid governor name, returns its ID.
 *
 * @kbdev: The &struct kbase_device for the GPU.
 * @name:  A string contrining the name of the governor.
 *
 * Return: the &enum gpu_dvfs_governor_type for @name. If not found, returns
 *         &GPU_DVFS_GOVERNOR_INVALID.
 */
enum gpu_dvfs_governor_type gpu_dvfs_governor_get_id(const char *name)
{
	int i;

	/* We use sysfs_streq here as name may be a sysfs input string */
	for (i = 0; i < GPU_DVFS_GOVERNOR_COUNT; i++)
		if (sysfs_streq(name, governors[i].name))
			return governors[i].id;

	return GPU_DVFS_GOVERNOR_INVALID;
}

/**
 * gpu_dvfs_governor_print_available() - Prints the names of the available governors.
 *
 * @buf:  The memory region to write out the governor names to.
 * @size: The maximum amount of data to write into @buf.
 *
 * Return: The amount of chars written to @buf.
 */
ssize_t gpu_dvfs_governor_print_available(char *buf, ssize_t size)
{
	int i;
	ssize_t ret = 0;

	for (i = 0; i < GPU_DVFS_GOVERNOR_COUNT; i++)
		ret += scnprintf(buf + ret, size - ret, "%s ", governors[i].name);

	ret += scnprintf(buf + ret, size - ret, "\n");

	return ret;
}

/**
 * gpu_dvfs_governor_print_curr() - Prints the name of the current governor.
 *
 * @kbdev: The &struct kbase_device for the GPU.
 * @buf:  The memory region to write out the name to.
 * @size: The maximum amount of data to write into @buf.
 *
 * Return: The amount of chars written to @buf.
 */
ssize_t gpu_dvfs_governor_print_curr(struct kbase_device *kbdev, char *buf, ssize_t size)
{
	struct pixel_context *pc = kbdev->platform_context;

	return scnprintf(buf, size, "%s\n", governors[pc->dvfs.governor.curr].name);
}

/**
 * gpu_dvfs_governor_init() - Initializes the Pixel GPU DVFS governor subsystem.
 *
 * @kbdev: The &struct kbase_device for the GPU.
 *
 * Return: On success, returns 0. Currently only returns success.
 */
int gpu_dvfs_governor_init(struct kbase_device *kbdev)
{
	const char *governor_name;

	struct pixel_context *pc = kbdev->platform_context;
	struct device_node *np = kbdev->dev->of_node;

	if (of_property_read_string(np, "gpu_dvfs_governor", &governor_name)) {
		GPU_LOG(LOG_WARN, kbdev, "GPU DVFS governor not specified in DT, using default\n");
		pc->dvfs.governor.curr = GPU_DVFS_GOVERNOR_BASIC;
		goto done;
	}

	pc->dvfs.governor.curr = gpu_dvfs_governor_get_id(governor_name);
	if (pc->dvfs.governor.curr == GPU_DVFS_GOVERNOR_INVALID) {
		GPU_LOG(LOG_WARN, kbdev, "GPU DVFS governor \"%s\" doesn't exist, using default\n",
			governor_name);
		pc->dvfs.governor.curr = GPU_DVFS_GOVERNOR_BASIC;
		goto done;
	}

done:
	return 0;
}

/**
 * gpu_dvfs_governor_term() - Terminates the Pixel GPU DVFS QOS subsystem.
 *
 * @kbdev: The &struct kbase_device for the GPU.
 *
 * Note that this function currently doesn't do anything.
 */
void gpu_dvfs_governor_term(struct kbase_device *kbdev)
{
}