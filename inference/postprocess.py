from typing import List

def smooth_predictions(
    preds: List[float], 
    smooth_num_frames: int,
    hangover: bool, 
    min_speech_frames: int
) -> List[float]:

    """
        Returns smoothed model predictions.

        :param preds: List[float], a list of predictions with values in {0.0, 1.0}
        :param smooth_num_frames: int, number of frames to use for smoothing
        :param min_speech_frames: int, minimum numer of consecutive frames for a speech segment
        :return: List[float], a list of smoothed predictions with values in {0.0, 1.0}
    """
    
    # Smoothing
    smoothed_preds = []
    for i in range(smooth_num_frames - 1, len(preds), smooth_num_frames):
        cur_pred = preds[i]
        if cur_pred == preds[i - 1] == preds[i - 2]:
            smoothed_preds.extend(smooth_num_frames * [cur_pred])
        else:
            if len(smoothed_preds) > 0:
                smoothed_preds.extend(smooth_num_frames * [smoothed_preds[-1]])
            else:
                smoothed_preds.extend(smooth_num_frames * [0.0])
    
    if len(smoothed_preds) != len(preds):
        smoothed_preds = (smooth_num_frames - 1) * [0.0] + smoothed_preds
    
    # Hangover (delayed transition from speech to non-speech)
    if hangover:
        n = 0
        while n < len(smoothed_preds):
            cur_pred = smoothed_preds[n]
            if cur_pred == 1.0:
                if n > 0:
                    smoothed_preds[n - 1] = 1.0
                if n < len(smoothed_preds) - 1:
                    smoothed_preds[n + 1] = 1.0
                n += 2
            else:
                n += 1

    # Min consecutive speech frames
    seg_len = 0
    for i, v in enumerate(smoothed_preds):
        if v == 1.0:
            seg_len += 1
            if i == len(smoothed_preds)-1:
                for j in range(i-seg_len, i+1):
                    smoothed_preds[j] = 0.0
        if v == 0.0:
            if seg_len < min_speech_frames:
                for j in range(i-seg_len, i):
                    smoothed_preds[j] = 0.0
            seg_len = 0
    
    return smoothed_preds
