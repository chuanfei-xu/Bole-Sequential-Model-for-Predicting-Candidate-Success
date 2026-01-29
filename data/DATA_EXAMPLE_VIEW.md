# Data Example (Simplified View)

This is a simplified view of the data structure for reference. **The actual data should be in JSON format**.

## Sample 1 - Failed at First Interview (Label: 100)

| Field | Value (Truncated) |
|-------|-------------------|
| application_id | APP_1 |
| resume_id | 123456 |
| talent_label | 100 |
| work_exp | 时间:2015.07-至今, 公司:某大型快消品公司, 职位:区域销售经理... |
| project_exp | 时间:2018.09-2019.03, 项目:渠道下沉新品推广项目... |
| edu_exp | 时间:2011.09-2015.07, 学校:某省重点大学, 学历:bachelor, 专业:市场营销 |
| post_text | 岗位名称：新零售区域运营经理... |
| pre_eval_comm | 执行能力强，过往业绩达成稳定... |
| eval_comm | 候选人有8年快消行业经验... |
| intv_1_comm | 初面反馈：候选人对快消行业有清晰见解... |
| intv_2_comm | (empty - failed at stage 2) |
| talent_cate_name_l1_merge | 市场,销售类 |
| post_cate_name_l1_merge | 市场 |
| data_type | train |

---

## Sample 2 - Passed All Stages (Label: 111)

| Field | Value (Truncated) |
|-------|-------------------|
| application_id | APP_2 |
| resume_id | 654321 |
| talent_label | 111 |
| work_exp | 时间:2018.07-至今, 公司:某知名互联网公司, 职位:算法工程师... |
| project_exp | 时间:2022.01-2022.09, 项目:多模态推荐算法优化... |
| edu_exp | 时间:2015.09-2018.07, 学校:某985高校, 学历:master, 专业:计算机科学与技术 |
| post_text | 岗位名称：推荐算法专家... |
| pre_eval_comm | 技术能力突出，项目经验契合岗位需求... |
| eval_comm | 候选人技术背景扎实，算法能力强... |
| intv_1_comm | 初面反馈：算法基础非常扎实... |
| intv_2_comm | 终面反馈：技术深度和广度均符合要求... |
| talent_cate_name_l1_merge | 技术类 |
| post_cate_name_l1_merge | 技术类 |
| data_type | train |

---

