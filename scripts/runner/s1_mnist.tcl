# STEP 1: Open session
open_session tenstorrent_bh mnist 4

# STEP 2: Set compilation recipe
set_core_group_shape  4 4
set_core_group_offset 0 0

set_session_recipe main_space_size_per_channel      1GB
set_session_recipe data_space_size_per_core         1MB
set_session_recipe data_space_size_per_core         512KB

set_session_recipe broadcast_optimize_queue_depth   8
set_session_recipe broadcast_optimize_max_ref_cnt   4
set_session_recipe context_buffer_slot_num          16
set_session_recipe ld_ex_buffer_slot_num            16
set_session_recipe ex_st_buffer_slot_num            8
set_session_recipe concurrent_load_num              1

set_session_recipe greedy_temporal_reuse            true

set_session_recipe temporal_reuse_type              ALL
set_session_recipe spatial_reuse_type               SINGLE_MAIN

set_session_recipe dtype                            float16
set_session_recipe acc_dtype                        float16

# STEP 3: Compile and run graph
mkdir ./output

enable_monitoring                   # make sure you have already run the monitoring server (neuromta_monitor_server)
enable_profiler ./output/profiles   # note that this will generate a large amount of profiling data, so specify an appropriate output path

compile_graph
run_graph

# STEP 4: Save results and exit
save_results ./output/results.json 
save_compile_summary ./output/compile_summary.json

exit