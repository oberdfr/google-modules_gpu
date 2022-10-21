/*
 * This program is free software and is provided to you under the terms of the
 * GNU General Public License version 2 as published by the Free Software
 * Foundation, and any use by you of this program is subject to the terms
 * of such GNU licence.
 *
 * A copy of the licence is included with the program, and can also be obtained
 * from Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#include <linux/mman.h>
#include <mali_kbase.h>

/* mali_kbase_mmap.c
 *
 * This file contains Linux specific implementation of
 * kbase_context_get_unmapped_area() interface.
 */


/**
 * align_and_check() - Align the specified pointer to the provided alignment and
 *                     check that it is still in range.
 * @gap_end:        Highest possible start address for allocation (end of gap in
 *                  address space)
 * @info:           vm_unmapped_area_info structure passed to caller, containing
 *                  alignment, length and limits for the allocation
 * @is_shader_code: True if the allocation is for shader code (which has
 *                  additional alignment requirements)
 * @is_same_4gb_page: True if the allocation needs to reside completely within
 *                    a 4GB chunk
 *
 * Return: true if gap_end is now aligned correctly and is still in range,
 *         false otherwise
 */
static bool align_and_check(unsigned long *gap_end,
		struct vm_unmapped_area_info *info, bool is_shader_code,
		bool is_same_4gb_page)
{
	/* Computing highest address at desired alignment is already handled
	 * by unmapped_area_topdown() via VM_UNMAPPED_AREA_TOPDOWN */

	if (is_shader_code) {
		/* Check for 4GB boundary */
		if (0 == (*gap_end & BASE_MEM_MASK_4GB))
			(*gap_end) -= (info->align_offset ? info->align_offset :
					info->length);
		if (0 == ((*gap_end + info->length) & BASE_MEM_MASK_4GB))
			(*gap_end) -= (info->align_offset ? info->align_offset :
					info->length);

		if (!(*gap_end & BASE_MEM_MASK_4GB) || !((*gap_end +
				info->length) & BASE_MEM_MASK_4GB))
			return false;
	} else if (is_same_4gb_page) {
		unsigned long start = *gap_end;
		unsigned long end = *gap_end + info->length;
		unsigned long mask = ~((unsigned long)U32_MAX);

		/* Check if 4GB boundary is straddled */
		if ((start & mask) != ((end - 1) & mask)) {
			unsigned long offset = end - (end & mask);
			/* This is to ensure that alignment doesn't get
			 * disturbed in an attempt to prevent straddling at
			 * 4GB boundary. The GPU VA is aligned to 2MB when the
			 * allocation size is > 2MB and there is enough CPU &
			 * GPU virtual space.
			 */
			unsigned long rounded_offset =
					ALIGN(offset, info->align_mask + 1);

			start -= rounded_offset;
			end -= rounded_offset;

			*gap_end = start;

			/* The preceding 4GB boundary shall not get straddled,
			 * even after accounting for the alignment, as the
			 * size of allocation is limited to 4GB and the initial
			 * start location was already aligned.
			 */
			WARN_ON((start & mask) != ((end - 1) & mask));
		}
	}


	if (*gap_end < info->low_limit)
		return false;

	return true;
}

/* This function is based on Linux kernel's arch_get_unmapped_area, but
 * simplified slightly. Modifications come from the fact that some values
 * about the memory area are known in advance.
 */
unsigned long kbase_context_get_unmapped_area(struct kbase_context *const kctx,
		const unsigned long addr, const unsigned long len,
		const unsigned long pgoff, const unsigned long flags)
{
	struct mm_struct *mm = current->mm;
	struct vm_unmapped_area_info info;
	unsigned long align_offset = 0;
	unsigned long align_mask = 0;
	unsigned long high_limit = mm->mmap_base;
	unsigned long low_limit = PAGE_SIZE;
	int cpu_va_bits = BITS_PER_LONG;
	int gpu_pc_bits =
	      kctx->kbdev->gpu_props.props.core_props.log2_program_counter_size;
	bool is_shader_code = false;
	bool is_same_4gb_page = false;
	unsigned long ret;

	/* the 'nolock' form is used here:
	 * - the base_pfn of the SAME_VA zone does not change
	 * - in normal use, va_size_pages is constant once the first allocation
	 *   begins
	 *
	 * However, in abnormal use this function could be processing whilst
	 * another new zone is being setup in a different thread (e.g. to
	 * borrow part of the SAME_VA zone). In the worst case, this path may
	 * witness a higher SAME_VA end_pfn than the code setting up the new
	 * zone.
	 *
	 * This is safe because once we reach the main allocation functions,
	 * we'll see the updated SAME_VA end_pfn and will determine that there
	 * is no free region at the address found originally by too large a
	 * same_va_end_addr here, and will fail the allocation gracefully.
	 */
	struct kbase_reg_zone *zone =
		kbase_ctx_reg_zone_get_nolock(kctx, KBASE_REG_ZONE_SAME_VA);
	u64 same_va_end_addr = kbase_reg_zone_end_pfn(zone) << PAGE_SHIFT;

	/* err on fixed address */
	if ((flags & MAP_FIXED) || addr)
		return -EINVAL;

#if IS_ENABLED(CONFIG_64BIT)
	/* too big? */
	if (len > TASK_SIZE - SZ_2M)
		return -ENOMEM;

	if (!kbase_ctx_flag(kctx, KCTX_COMPAT)) {
		high_limit =
			min_t(unsigned long, mm->mmap_base, same_va_end_addr);

		/* If there's enough (> 33 bits) of GPU VA space, align
		 * to 2MB boundaries.
		 */
		if (kctx->kbdev->gpu_props.mmu.va_bits > 33) {
			if (len >= SZ_2M) {
				align_offset = SZ_2M;
				align_mask = SZ_2M - 1;
			}
		}

		low_limit = SZ_2M;
	} else {
		cpu_va_bits = 32;
	}
#endif /* CONFIG_64BIT */
	if ((PFN_DOWN(BASE_MEM_COOKIE_BASE) <= pgoff) &&
		(PFN_DOWN(BASE_MEM_FIRST_FREE_ADDRESS) > pgoff)) {
		int cookie = pgoff - PFN_DOWN(BASE_MEM_COOKIE_BASE);
		struct kbase_va_region *reg;

		/* Need to hold gpu vm lock when using reg */
		kbase_gpu_vm_lock(kctx);
		reg = kctx->pending_regions[cookie];
		if (!reg) {
			kbase_gpu_vm_unlock(kctx);
			return -EINVAL;
		}
		if (!(reg->flags & KBASE_REG_GPU_NX)) {
			if (cpu_va_bits > gpu_pc_bits) {
				align_offset = 1ULL << gpu_pc_bits;
				align_mask = align_offset - 1;
				is_shader_code = true;
			}
#if !MALI_USE_CSF
		} else if (reg->flags & KBASE_REG_TILER_ALIGN_TOP) {
			unsigned long extension_bytes =
				(unsigned long)(reg->extension
						<< PAGE_SHIFT);
			/* kbase_check_alloc_sizes() already satisfies
			 * these checks, but they're here to avoid
			 * maintenance hazards due to the assumptions
			 * involved
			 */
			WARN_ON(reg->extension >
				(ULONG_MAX >> PAGE_SHIFT));
			WARN_ON(reg->initial_commit > (ULONG_MAX >> PAGE_SHIFT));
			WARN_ON(!is_power_of_2(extension_bytes));
			align_mask = extension_bytes - 1;
			align_offset =
				extension_bytes -
				(reg->initial_commit << PAGE_SHIFT);
#endif /* !MALI_USE_CSF */
		} else if (reg->flags & KBASE_REG_GPU_VA_SAME_4GB_PAGE) {
			is_same_4gb_page = true;
		}
		kbase_gpu_vm_unlock(kctx);
#ifndef CONFIG_64BIT
	} else {
		return current->mm->get_unmapped_area(
			kctx->filp, addr, len, pgoff, flags);
#endif
	}

	info.flags = VM_UNMAPPED_AREA_TOPDOWN;
	info.length = len;
	info.low_limit = low_limit;
	info.high_limit = high_limit;
	info.align_offset = align_offset;
	info.align_mask = align_mask;

	while ((ret = vm_unmapped_area(&info)) > 0) {
		if (align_and_check(&ret, &info,
				is_shader_code, is_same_4gb_page))
			break;
		info.high_limit = ret;
	}

	if (IS_ERR_VALUE(ret) && high_limit == mm->mmap_base &&
	    high_limit < same_va_end_addr) {
		/* Retry above mmap_base */
		info.low_limit = mm->mmap_base;
		info.high_limit = min_t(u64, TASK_SIZE, same_va_end_addr);

		while ((ret = vm_unmapped_area(&info)) > 0) {
			if (align_and_check(&ret, &info,
					is_shader_code, is_same_4gb_page))
				break;
			info.high_limit = ret;
		}
	}

	VM_BUG_ON(ret != -ENOMEM);
	WARN_ON(IS_ERR_VALUE(ret));

	return ret;
}
