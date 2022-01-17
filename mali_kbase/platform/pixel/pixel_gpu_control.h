/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright 2020-2021 Google LLC.
 *
 * Author: Sidath Senanayake <sidaths@google.com>
 */

#ifndef _PIXEL_GPU_CONTROL_H_
#define _PIXEL_GPU_CONTROL_H_

/* Power management */
#ifdef CONFIG_MALI_PIXEL_GPU_PM
bool gpu_pm_get_power_state(struct kbase_device *kbdev);
int gpu_pm_init(struct kbase_device *kbdev);
void gpu_pm_term(struct kbase_device *kbdev);
#else
static bool __maybe_unused gpu_pm_get_power_state(struct kbase_device *kbdev) { return true; }
static int __maybe_unused gpu_pm_init(struct kbase_device *kbdev) { return 0; }
static void __maybe_unused gpu_pm_term(struct kbase_device *kbdev) {}
#endif

/* DVFS */
#ifdef CONFIG_MALI_MIDGARD_DVFS
void gpu_dvfs_event_power_on(struct kbase_device *kbdev);
void gpu_dvfs_event_power_off(struct kbase_device *kbdev);
int gpu_dvfs_init(struct kbase_device *kbdev);
void gpu_dvfs_term(struct kbase_device *kbdev);
#else
static int __maybe_unused gpu_dvfs_init(struct kbase_device *kbdev) { return 0; }
static void __maybe_unused gpu_dvfs_term(struct kbase_device *kbdev) {}
#endif

/* sysfs */
#ifdef CONFIG_MALI_MIDGARD_DVFS
int gpu_sysfs_init(struct kbase_device *kbdev);
void gpu_sysfs_term(struct kbase_device *kbdev);
#else
static int __maybe_unused gpu_sysfs_init(struct kbase_device *kbdev) { return 0; }
static void __maybe_unused gpu_sysfs_term(struct kbase_device *kbdev) {}
#endif

/* Kernel context callbacks */
#ifdef CONFIG_MALI_MIDGARD_DVFS
int gpu_dvfs_kctx_init(struct kbase_context *kctx);
void gpu_dvfs_kctx_term(struct kbase_context *kctx);
#endif

#endif /* _PIXEL_GPU_CONTROL_H_ */
