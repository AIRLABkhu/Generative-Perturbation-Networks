# Generative-Perturbation-Networks

# Title
Generative Perturbation Network for Signal-Agnostic/Specific Adversarial Attacks on EEG-based Brain-Computer Interfaces

# Abstract
Brain-computer interface (BCI) enables direct communication between a brain and computers or external devices. Recently, deep neural networks (DNNs) have
achieved great success in classifying EEG-based BCI systems. However, DNNs are vulnerable to adversarial attacks using adversarial examples imperceptible to humans. This paper proposes a novel generative model named generative perturbation network (GPN), able to generate adversarial examples by signal-agnostic and signal-specific perturbations for targeted and non-targeted attacks. By modifying the proposed model slightly, we can also generate conditional or total perturbations for an EEG dataset with a pretrained weight. Our experimental evaluation demonstrates that perturbations generated by the proposed model outperform previous approaches for crafting signal-agnostic perturbation in non-targeted and targeted attacks. Moreover, we show that modified models, i.e., conditional and multiple GPN, can generate perturbations of all classification models, attack types, and target classes with single trained parameters only. Finally, we show that the proposed method has higher transferability across classification networks than comparison methods, demonstrating the perturbations are well generalized.

# Architecture
## Generative Perturbation Networks (GPNs)
![image](https://user-images.githubusercontent.com/50229148/170931136-db19f146-4f73-40ab-8311-0ebbe36077b2.png)

- Signal-Specific generation : Input as a real EEG trials
- Signal-Agnoistic generation : Input as a random noise <br>
![image](https://user-images.githubusercontent.com/50229148/170930773-a2f59d1a-e5a1-469e-ad7c-b692d3264239.png)

## Conditional Generative Perturbation Network(cGPN)
- cGPN recieves bith a signal and condition vector as input for defining the perturbation
- cGPN generates perturbations for all types of classification model, attack type, and target classes for one dataset <br>
![image](https://user-images.githubusercontent.com/50229148/170931926-1a1e2b4c-053e-41c6-bd34-2349e692cf02.png)

## Multiple Generative Perturbations Network(mGPN)
- mGPN generates all perturbations for an input EEG trial at once. <br>
![image](https://user-images.githubusercontent.com/50229148/170931960-b8b89be5-5cec-4c73-8fa0-77a73b61f8ea.png)

# Result

## Accuray and Fooling Rate
- Accuracy : Top 1 accuracy of classification
- Fooling Rate : For non-targeted only. Prediction difference from original prediction

### Comparision of **DF-UAP, TLM, GPN-SA** methods
![image](https://user-images.githubusercontent.com/50229148/170942206-3fc924c8-a374-4cdc-93ac-4facceae3602.png)

### Comparision of **GPN-SS, cGPN, mGPN** methods
![image](https://user-images.githubusercontent.com/50229148/170942328-68eade0e-f9f1-4671-ae69-5ef77b245d98.png)

## Transferability
![image](https://user-images.githubusercontent.com/50229148/170943025-13ae6a14-285f-4a7b-a6ee-170bb6c33bcf.png)

## Topoplots of adversarial exampels from mGPN
![image](https://user-images.githubusercontent.com/50229148/170933337-6df1c5bc-a0c8-41c4-8298-2fe96ae6cb35.png)

# Usage
> Train
- `train_classifier.py` : Train victim models for EEG classification. EEGNET, Deep/Shallow ConvNet, TIDNET, VGG, ResNet were used as victim models.
- `train_uap_df.py` : Pytorch Implementation of *Universal adversarial perturbations, CVPR, 2017, Moosavi et al.* `Adversarial-Robustness-Toolbox` library was used for Deepfool based attacks.
- `train_uap_tlm.py` : Pytorch Implementation of *Universal adversarial perturbations for CNN classifiers in EEG-based BCIs, JOURNAL OF NEURAL ENGINEERING, 2021, Z.Liu et al.*<br>
The offical implementaion(Tensorflow ver.) can be found here: https://github.com/ZihanLiu95/UAP_EEG. 
- `train_GPN_SS.py` : Train GPN for Signal Specific perturbations generation.
- `train_GPN_SA.py` : Train GPN for Signal Agnositic perturbations generation.
- `train_cGPN.py` : Train conditional GPN(cGPN) for generating perturbations with conditional flags.
- `train_mGPN.py` : Train multiple GPN(mGPN) for generating perturbations at once.
> Evaluate
- `eval_classifier.py` : Evaluate accuacy of trained victim models for EEG classification.
- `eval_GPN_SS&SA.py` : Evaluate accuracy of adversarial attacks with GPN on victim models.
- `eval_GPN_SS&SA_cross.py` : Evaluate Transferability of Universal Perturbations.
> Model Architecture Implementation -> `./adversarial models'
- `GenResNet.py` : Implementations of Generative Perturbation Networks for SS and SA attacks
- `GenResNetHyper.py` : Implementations of conditional Generative Perturbation Networks
- `GenResNetMulti.py` : Implementations of multiple Generative Perturbation Networks
> Models for EEG classification -> `./models'
- `EEGNet.py` : Pytorch Implementaion of *EEGNet a compact convolutional neural network, JOURANL OF NEURAL ENGINEERING, 2018, V.Lawhern et al.* Origianl Implemenation: https://github.com/vlawhern/arl-eegmodels
- `DeepConvNet.py` : Pytorch Implementation of DeepConvNet. Origianl Implemenation: https://github.com/vlawhern/arl-eegmodels
- `ShallowConvNet.py`: Pytorch Implementation of ShallowConvNet
- `TIDNet.py` : Pytorch Implementaion of TIDNet
- `VGG.py` : Pytorch Implementaion of VGG
- `ResNet.py` : Pytorch Implementaion of ResNet
