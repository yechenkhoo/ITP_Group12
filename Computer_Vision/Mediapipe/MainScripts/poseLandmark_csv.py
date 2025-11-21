import os, cv2, glob,  argparse, math, re, numpy as np, pandas as pd, mediapipe as mp

def extract_frame_number(filename):
    match = re.search(r'frame(\d+)', filename)
    return int(match.group(1)) if match else -1

def extract_landmarks(dataset_path, save_path):

    ##############
    torso_size_multiplier = 2.5
    n_landmarks = 33
    n_dimensions = 3
    landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]
    ##############

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
        

    class_list = os.listdir(dataset_path)
    # class_list = sorted(class_list)

    col_names = []
    for i in range(n_landmarks):
        name = mp_pose.PoseLandmark(i).name
        name_x = name + '_X'
        name_y = name + '_Y'
        name_z = name + '_Z'
        name_v = name + '_V'
        col_names.append(name_x)
        col_names.append(name_y)
        col_names.append(name_z)
        col_names.append(name_v)

    full_lm_list = []
    full_lm_unnorm_list = []
    target_list = []
    image_path_list = []
    frame_name_list = []
    total_fail = 0

    for class_name in class_list:
        path_to_class = os.path.join(dataset_path, class_name)
        img_list = glob.glob(path_to_class + '/*.jpg') + glob.glob(path_to_class + '/*.jpeg')+ glob.glob(path_to_class + '/*.png')      
        img_list = sorted(img_list, key=lambda x: extract_frame_number(os.path.basename(x)))

        fail = 0 
        # Read reach Images in the each classes
        for img in img_list:
            image = cv2.imread(img)
            h, w, c = image.shape
            if image is None:
                print(
                    f'[ERROR] Error in reading {img} -- Skipping.....\n[INFO] Taking next Image')
                continue
            else:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8) # New line added to curb bugs arisen from library changes
                result = pose.process(img_rgb)
                if result.pose_landmarks:
                    lm_list = []
                    for landmarks in result.pose_landmarks.landmark:
                        # Preprocessing
                        max_distance = 0
                        lm_list.append(landmarks)
                    pre_lm_unnorm = list(np.array([
                        [landmark.x * w, landmark.y * h, landmark.z * w, landmark.visibility]
                        for landmark in lm_list
                    ]).flatten())
                    full_lm_unnorm_list.append(pre_lm_unnorm)

                    center_x = (lm_list[landmark_names.index('right_hip')].x +
                                lm_list[landmark_names.index('left_hip')].x)*0.5
                    center_y = (lm_list[landmark_names.index('right_hip')].y +
                                lm_list[landmark_names.index('left_hip')].y)*0.5

                    shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                                lm_list[landmark_names.index('left_shoulder')].x)*0.5
                    shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                                lm_list[landmark_names.index('left_shoulder')].y)*0.5

                    for lm in lm_list:
                        distance = math.sqrt(
                            (lm.x - center_x)**2 + (lm.y - center_y)**2)
                        if(distance > max_distance):
                            max_distance = distance
                    torso_size = math.sqrt(
                        (shoulders_x - center_x)**2 + (shoulders_y - center_y)**2)
                    max_distance = max(
                        torso_size*torso_size_multiplier, max_distance)

                    pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, (landmark.y-center_y)/max_distance,
                                landmark.z/max_distance, landmark.visibility] for landmark in lm_list]).flatten())

                    full_lm_list.append(pre_lm)
                    target_list.append(class_name)

                    rel_path = os.path.relpath(img, start=dataset_path)
                    image_path_list.append(rel_path.replace('\\', '/'))  # use forward slashes for consistency
                    frame_name = os.path.basename(img)
                    frame_name_list.append(frame_name)
                    # print(f'{os.path.split(img)[1]} Landmarks added Successfully')
                else:
                    print(f'[WARNING] No pose landmarks detected in {img} -- Skipping...')
                    fail += 1
                    total_fail += 1

                    # save failed frames into csv as well
                    empty_lm = [np.nan] * len(col_names)
                    full_lm_list.append(empty_lm)
                    full_lm_unnorm_list.append(empty_lm)
                    target_list.append(class_name)

                    rel_path = os.path.relpath(img, start=dataset_path)
                    image_path_list.append(rel_path.replace('\\', '/'))
                    frame_name_list.append(os.path.basename(img))
                    # continue
                
        print(f'[INFO] {class_name} Successfully Completed')
        print(f'[WARNING] {fail} Failed')

    print('[INFO] Landmarks from Dataset Successfully Completed')
    print(f'[WARNING] {total_fail} total failed.')

    data_x = pd.DataFrame(full_lm_list, columns=col_names)
    data = pd.DataFrame({"Frame_Name": frame_name_list})

    data = pd.concat([data, data_x], axis=1)

    data.to_csv(save_path, encoding='utf-8', index=False)
    print(f'[INFO] Successfully Saved Landmarks data into {save_path}')

    unnorm_df = pd.DataFrame(full_lm_unnorm_list, columns=col_names)
    unnorm_df.insert(0, "Frame_Name", frame_name_list)
    base, ext = os.path.splitext(save_path)
    unnorm_path = f"{base}_unnormalised{ext}"
    unnorm_df.to_csv(unnorm_path, encoding='utf-8', index=False)
    print(f"[INFO] Also saved unnormalized landmarks → {unnorm_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", type=str, required=True,
                    help="path to dataset/dir")
    ap.add_argument("-o", "--save", type=str, required=True,
                    help="path to save csv file, eg: dir/data.csv")
    args = vars(ap.parse_args())

    extract_landmarks(args["dataset"], args["save"])
