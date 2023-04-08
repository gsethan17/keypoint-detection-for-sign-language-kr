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
