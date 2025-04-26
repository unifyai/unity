# Learning From Video

1. Detect all candidate "Action" timestamps from the video, using some off-the-shelf video action detection model

2. For each adjacent pair of screenshots (either side of the action timestamp), pass the pair of images, the overall task description (if available) and any overlapping transcript (if available), all through a multi-modal LLM.  Ask if an action was performed (structured output with a boolean field), and if so, what would be a good description of the action (structured output, optional str field)

3. Pass all extracted (timestamped) action descriptions, the (timestamped) transcript (if available) as well as some of the screenshots (all of them is probably too expensive) to an LLM, and ask it to generate an overall description of the full task (or append the existing description if one was provided).

4. Iteratively prompt the model to merge actions which it deems are semantically identical (ie "click LinkedIn homepage" and "Click on LinkedIn homepage"). Do this until it is happy that no more actions can be merged.

5. Use this overall task description and the (now reduced) series of **semantically distinct** action desriptions (with several actions now repeated throughout the flat action trace) + the transcript (timestamped) to generate a plan for this task, implemented in Python code, in a zero-shot manner using a SOTA model (such as `o3`), inform the agent that the trace of low-level actions must map **exactly** to the trace extracted from the video. Continue looping until the generated code matches the low-level command trace exactly (ideally some higher level patterns have been detected, with nested function calls in the generated code).

6. Start to execute this code using the `Planner`, implemented [like this](https://github.com/unifyai/unity/blob/12e379b7e535a03168c436c29f0c7d90e2399292/planner/planner_proposal.py), and incorporate cross-referencing against the original screenshots during the `_verify_completes` method, to verify that the live task execution does indeed match the observations which were seen in the original video. If not, then `_verify_completes` will continue trying new implementations for that function until they do match. This happens for every function in the source code, both low-level and high-level, with the option to repeat the current function or step up the stack to the parent function to modify it's own source code, if the higher-level pattern is actually wrong in the first place.

7. Once this process finishes, there will be a python implementation which ideally includes nested functions (and therefore abtract concepts), which should hopefully generalize to new unseen instances of the same task with slight variations (ie we haven't just blindly copied a flat list of low-level actions, we have captured broader structure, expressed through the higher level functions). We will also have verified that the task was performed in a manner which was visually the same as in the demo video, as every function ended up passing the verification check against the source video (otherwise, the task extraction process is marked as failed).

8. Provided it did not fail, we now have a repeatable task, learned entirely from a single demo video!
