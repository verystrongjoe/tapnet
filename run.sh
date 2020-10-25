python train.py --dataset StandWalkJump --rp_param 3,4 --lr 1e-4
python train.py --dataset Cricket --lr 1e-4
python train.py --dataset ArticularyWordRecognition --dilation 5 --lr 1e-4
python train.py --dataset NATPOS --lr 1e-4
python train.py --dataset PEMS-SF --lr 1e-4
python train.py --dataset BasicMotions --lr 1e-4
python train.py --dataset Libras --lr 1e-4
python train.py --dataset PenDigits --kernels 4,1,1 --lr 1e-4   
python train.py --dataset HandMovementDirection --lr 1e-3
python train.py --dataset JapaneseVowels  --lr 1e-3 --kernels 3,2,2
python train.py --dataset AtrialFibrilation --lr 1e-4  --kernels 3,1,1
python train.py --dataset RacketSports --lr 1e-4 --kernels 3,2,2
python train.py --dataset FingerMovements --lr 1e-4
python train.py --dataset LSST --lr 1e-3
python train.py --dataset Heartbeat --lr 1e-4 --dilation 2
python train.py --dataset SpokenArabicDigits --lr 1e-4 --filters 64,64,32 
python train.py --dataset SelfRegulationSCP2 --filters 64,64,32 —lr 1e-6 
python train.py --dataset Phoneme --filters 64,64,32 --lr 1e-4
python train.py --dataset FaceDetection --dilation 20 --kernels 4,1,1 --filters 32,32,16 --lr 1e-3
python train.py --dataset MotorImagery --dilation 200 --filters 64,64,32 --lr 1e-4
python train.py --dataset DuckDuckGeese --lr 1e-3
python train.py --dataset EthanolConcentration  --rp_param 3,1 --dilation 100 --filters 64,64,64 --lr 1e-4
python train.py --dataset Epilepsy --wd 1e-2 --lr 1e-6 
python train.py --dataset UWaveGestureLibrary --lr 1e-4
python train.py --dataset SelfRegulationSCP1 --lr 1e-7 --dilation 50
python train.py --dataset EigenWorms --dilation 2000 --filters 64,64,32 --lr 1e-4
python train.py --dataset CharacterTrajectories --no-cuda --lr 1e-4
python train.py --dataset ERing --lr 1e-4
python train.py --dataset Handwriting --lr 1e-4
python train.py --dataset InsectWingbeat --filters 4,8,4 --dilation 10 --no-cuda --lstm_dim 32
