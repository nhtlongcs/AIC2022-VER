def get_frame_ids_by_names(frame_names):
    """
    Get frame id by its names
    """
    frame_ids = [
        int(i.split('/')[-1][:-4])
        for i in frame_names
    ]
    return frame_ids