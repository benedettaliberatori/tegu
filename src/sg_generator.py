import re
import os
import json
from tqdm import tqdm
from models.llavaov_model import LLaVAOneVisionModel
from data.video_dataset import VideoDataset
from torch.utils.data import DataLoader
from utils import set_all_seeds


###################################################################################################################à
# prompt = """
#     Describe the video in detail. Then, based on the description, generate scene graph triplets in the form of (subject, predicate, object).
#     Respond in the following JSON format (without any explanation or additional text):
#     {
#       "description": "<detailed description of the video>",
#       "triplets": [
#         ["<subject1>", "<predicate1>", "<object1>"],
#         ["<subject2>", "<predicate2>", "<object2>"]
#       ]
#     }
#     """
# def parse_answer(answer):
#     try:
#         # Extract the first valid JSON block using regex
#         match = re.search(r"{.*}", answer, re.DOTALL)
#         if match:
#             json_str = match.group(0)
#             data = json.loads(json_str)

#             triplets = data.get("triplets", [])
#             triplets = [
#                 " ".join(triplet).strip("() \n")
#                 for triplet in triplets
#                 if isinstance(triplet, list)
#             ]

#             return {"description": data.get("description", ""), "triplets": triplets}

#     except json.JSONDecodeError:
#         pass

#     return {"description": "", "triplets": []}


#########################################ààà previous JSON-based approach #########################################

def parse_answer(answer):
    parts = re.split(
        r'Scene Graph Triplets:|The scene graph triplets are as follows:|The scene graph triplets for this video are as follows:',
        answer, flags=re.IGNORECASE
    )    

    if len(parts) < 2:
        return {"description": answer.strip(), "triplets": []}

    description = parts[0].strip()
    triplet_part = parts[1].strip()

    try:
        pattern = r'\d+\.\s*\((.*?)\)'
        matches = re.findall(pattern, triplet_part)
        
        triplets = [' '.join(part.strip() for part in triplet.split(',')) for triplet in matches]

        return {
            "description": description,
            "triplets": triplets
        }
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return {"description": description, "triplets": []}

def main():

    set_all_seeds(42)
    verbose = False
    video_dir = "/data/shared/bliberatori/Thumos14/videos"
    fps = 1
    num_frames_per_clip = 8
    max_new_tokens = 512
    chunk_size_per_video = 30
    dataset = VideoDataset(video_dir, fps, num_frames_per_clip)
    dataset.video_list = [item for item in dataset.video_list if not os.path.exists(f"/gpfs/projects/ehpc160/bliberatori/tegu/src/generated_triplets/thumos/{item.split('/')[-1].split('.')[0]}.json")]
    dataset.video_list = dataset.video_list[::-1]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model = LLaVAOneVisionModel()
    prompt = "Describe the video in details. Based on the description, generate scene graph triplets in the form of (subject, predicate, object) based on the description. Answer with both the description and the sene graph triplets. The answer is expected to be in the format: <description> Scene Graph Triplets: 1. <triplet1> \n, 2. <triplet2> \n, ..."

    for video_batch in tqdm(dataloader):
        video_clips = video_batch["video_frames"]
        video_id = video_batch["video_id"][0]
        if verbose:
            print(f"Processing video ID: {video_id}")
            print(len(video_clips), "clips loaded for video:", video_id)

        video_clips_chunked = [
            video_clips[i : i + chunk_size_per_video]
            for i in range(0, len(video_clips), chunk_size_per_video)
        ]  # video clips divided into chunks (list of lists)
        if verbose:
            print(f"Number of chunks: {len(video_clips_chunked)}")

        current_generated_triplets = {}
        clip_index = 0

        for i, clip in enumerate(video_clips_chunked):
            if verbose:
                print(
                    f"Processing chunk {i + 1}/{len(video_clips_chunked)} with {len(clip)} clips"
                )
            answers = model.generate(data=clip, prompt=prompt, max_new_tokens=max_new_tokens)

            for answer in answers:
                parsed = parse_answer(answer)  # see parser below
                
                current_generated_triplets[str(clip_index)] = {
                    "description": parsed["description"],
                    "triplets": parsed["triplets"],
                }
                if current_generated_triplets[str(clip_index)]["triplets"] == []:
                    print(
                        f"[WARNING] No triplets generated for clip {clip_index} in video {video_id}"
                    )
                clip_index += 1
        with open(f"generated_triplets/thumos/{video_id}.json", "w") as f:
            json.dump(current_generated_triplets, f, indent=4)

    return


if __name__ == "__main__":

    main()
