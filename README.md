# keypoint-detection-for-sign-language-kr  

You can get the [sample data](./data/sample_data) from the [AIHub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=103)

## openpifpaf
### install
When I install openpifpaf, I used commends as belows.
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html  # install pytorch
pip3 install openpifpaf==0.12.14  # install openpifpaf
```

### dependencies
When I predict keypoints from the video, I used commends before prediction.
```
pip3 install opencv-python  # install opencv
conda update ffmpeg  # update ffmpeg
```

### prediction commends
#### prediction single image
`python -m openpifpaf.predict ./27305_41109_79.jpg --image-output --json-output`
#### prediction single image to json file
`python -m json.tool ./27305_41109_79.jpg.predictions.json`
#### prediction video as body points
`python3 -m openpifpaf.video --source ./sample_video.mp4 --video-output --json-output`
#### prediction video as wholebody points
`python3 -m openpifpaf.video --source ./sample_video.mp4 --video-output --json-output --checkpoint=shufflenetv2k16-wholebody`
