/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 *
 * (C) COPYRIGHT 2020-2022 ARM Limited. All rights reserved.
 *
 * This program is free software and is provided to you under the terms of the
 * GNU General Public License version 2 as published by the Free Software
 * Foundation, and any use by you of this program is subject to the terms
 * of such GNU license.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-2.0.html.
 *
 */

#ifndef _KBASE_CSF_FIRMWARE_CFG_H_
#define _KBASE_CSF_FIRMWARE_CFG_H_

#include <mali_kbase.h>
#include "mali_kbase_csf_firmware.h"
#include <linux/firmware.h>

#define CONFIGURATION_ENTRY_NAME_OFFSET (0xC)

/**
 * kbase_csf_firmware_cfg_init - Create the sysfs directory for configuration
 *                               options present in firmware image.
 *
 * @kbdev: Pointer to the Kbase device
 *
 * This function would create a sysfs directory and populate it with a
 * sub-directory, that would contain a file per attribute, for every
 * configuration option parsed from firmware image.
 *
 * Return: The initialization error code.
 */
int kbase_csf_firmware_cfg_init(struct kbase_device *kbdev);

/**
 * kbase_csf_firmware_cfg_term - Delete the sysfs directory that was created
 *                               for firmware configuration options.
 *
 * @kbdev: Pointer to the Kbase device
 *
 */
void kbase_csf_firmware_cfg_term(struct kbase_device *kbdev);

/**
 * kbase_csf_firmware_cfg_option_entry_parse() - Process a
 *                                               "configuration option" section.
 *
 * @kbdev:     Kbase device structure
 * @fw:        Firmware image containing the section
 * @entry:     Pointer to the section
 * @size:      Size (in bytes) of the section
 * @updatable: Indicates if entry can be updated with FIRMWARE_CONFIG_UPDATE
 *
 * Read a "configuration option" section adding it to the
 * kbase_device:csf.firmware_config list.
 *
 * Return: 0 if successful, negative error code on failure
 */
int kbase_csf_firmware_cfg_option_entry_parse(struct kbase_device *kbdev,
					      const struct kbase_csf_mcu_fw *const fw,
					      const u32 *entry, unsigned int size, bool updatable);

#ifdef CONFIG_MALI_HOST_CONTROLS_SC_RAILS
/**
 * kbase_csf_firmware_cfg_enable_host_ctrl_sc_rails() - Enable the config in FW to support
 *                                                      Host based control of SC power rails
 *
 * Look for the config entry that enables support in FW for the Host based
 * control of shader core power rails and set it before the intial boot
 * or reload of firmware.
 *
 * @kbdev:     Kbase device structure
 *
 * Return: 0 if successful, negative error code on failure
 */
int kbase_csf_firmware_cfg_enable_host_ctrl_sc_rails(struct kbase_device *kbdev);
#endif

/**
 * kbase_csf_firmware_cfg_find_config_address() - Get a FW config option address
 *
 * @kbdev:     Kbase device structure
 * @name:      Name of cfg option to find
 * @addr:      Pointer to store the address
 *
 * Return: 0 if successful, negative error code on failure
 */
int kbase_csf_firmware_cfg_find_config_address(struct kbase_device *kbdev, const char *name,
					       u32 *addr);

#endif /* _KBASE_CSF_FIRMWARE_CFG_H_ */
