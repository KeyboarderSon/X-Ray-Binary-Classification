## Capstone design 1 (2)
### 심장비대증 판단을 위한  X-Ray Binary Classification

NIH dataset을 densenet으로 train한 뒤 이를 전이학습에 사용하여 binary classification을 수행합니다.  


Dataset의 구조는 normal, abnormal 폴더의 X-Ray 사진으로 구성되어 있습니다.  

test만 원한다면  
```TEST_MODEL = True```  
로 변경해야 합니다. 이때 모델구조와 weight가 저장된 .h5 파일이 필요합니다.