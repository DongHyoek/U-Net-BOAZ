# U-Net CODE H/W

## Dataset
Dataset은 지난번에 말씀드렸던 EM segmentation challenge에서 사용되었던 dataset입니다.
`image shape = (1,512,512)` : GRAY SCALE 이미지

데이터는 아래와 같이 구성됩니다.
Train : 24장 , Valid : 3 , Test : 3

## 순서

1) `requriements.txt`로 가상환경 생성

2) `data_preprocess.py`로 .tif형태의 파일을 전처리해줍니다.

3) `main.py`로 모델 학습 진행

4) `main.py --mode test`로 평가 진행

## Task

과제는 model.py에서만 진행하시면 됩니다.

### Task 1
해당 모델은 기존의 Unet과 달리 572 x 572 이미지가 아닌 **512 x 512가 Input으로 들어가고, Output도 512 x 512로** 나오도록 재구성되었습니다.

- Q) 그렇다면 어느 부분이 변경되어서 Output size가 Input Size와 동일하게 나왔는지 찾아주세요.

### Task 2
`modle.py`에서 주석으로 남겨진 빈칸 부분을 채워주세요! , 참고로 EM Segmentation Dataset은 **Class가 2개**입니다.

**위의 과제들을 다 진행해주시고 평가가 완료된 사진과 `./result` 폴더에서 Output 한장을 노션에 올려주세요**

#### 학습 및 평가 완료 예시 사진
![Alt text](image.png) ![Alt text](image-1.png)

#### Segmentation Output 예시 사진
![Alt text](output_0000.png)

