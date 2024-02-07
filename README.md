# Virtual Piano with OpenCV and Mediapipe

## Overview

This project implements a virtual piano using OpenCV and Mediapipe, allowing users to play music by interacting with their hands in front of a camera. The application detects hand keypoints using Mediapipe's Hand module and maps them to corresponding piano keys on the screen.

## Demo Video
[Virtual Piano Demo](demo/final_demo.mp4)
<!-- https://github.com/NAGAMALLYSRUJAN2329/vir_piano/assets/118573078/ab7de956-1eed-4572-b31f-2a1fe92d8d00 -->
https://github.com/NAGAMALLYSRUJAN2329/vir_piano/assets/129930232/7337adc2-70d1-4739-9d63-76f7401ebf8c

## Getting Started

### Installation

- Clone the repository
```
git clone https://github.com/NAGAMALLYSRUJAN2329/vir_piano.git
cd vir_piano
```
- Create new conda environment
```
conda create -n VirPiano python=3.10
```
- Activate the environment
```
conda activate VirPiano
```

- Install dependencies
```
pip install -r requirements.txt
```

#### Local setup

```
python main.py
```
- If you want to change settings of the paino, use the below cli command.
```
python main.py --model_path "model/hand_landmarker.task" --num_octaves 2 --list_of_octaves "[3,4]" --height_and_width_black "[[5,8],[5,8]]" --shape "(800,600,3)" --tap_threshold 20 --piano_config_threshold 30 --piano_config 1
```


##### Controls
- **Quit**: Press 'Q' key.

#### For Streamlit Interface

```
streamlit run streamlit_main.py
```



## Troubleshooting
- If there is no hand tracking, try adjusting the camera settings and lighting conditions and try to move the hand slowly in front of the camera.
- Ensure that your Python environment meets the specified requirements.

## Contributers
[NAGAMALLYSRUJAN2329](https://github.com/NAGAMALLYSRUJAN2329)
<br/>
[Prachi-0205](https://github.com/Prachi-0205)
<br/>
[Prkhar05](https://github.com/Prkhar05)
<br/>
[xenom2004](https://github.com/xenom2004)





## Contributing
Contributions are welcome!



