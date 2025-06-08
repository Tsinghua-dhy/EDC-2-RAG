python run_ours_ddtag_for_ddtags_dynamic.py 0518 full GPT4omini_request "[5,10,20,30,50,70,100]" "[0]" dynamic webq &
python run_ours_ddtag_for_ddtags_dynamic.py 0519_3 full GPT4omini_request "[20,100]" "[20,40,60,80,100]" dynamic twowiki &
python run_ours_ddtag_for_ddtags_dynamic.py 0519_3 full GPT4omini_request "[5,10,20,30,50,70,100]" "[0]" dynamic twowiki &
python run_all.py 0519_1 full GPT4omini_request "[5,10,20,30,50,70,100]" "[0]" 1110 dynamic twowiki &
python run_all.py 0519_1 full GPT4omini_request "[20,100]" "[20,40,60,80,100]" 1110 dynamic twowiki &
python run_baseline_wo_retrieve.py ChatGPT_request 0520 full twowiki &
python run_baseline_wo_retrieve.py ChatGPT_request 0520 full webq &
python run_baseline_wo_retrieve.py ChatGPT_request 0520 full musique &
python run_baseline_wo_retrieve.py GPT4omini_request 0520 full webq &
python run_baseline_wo_retrieve.py GPT4omini_request 0520 full twowiki &
python run_baseline_wo_retrieve.py GPT4omini_request 0520 full musique &
wait  # 等待所有进程结束
