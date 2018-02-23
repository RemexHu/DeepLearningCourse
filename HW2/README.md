# HW2

### Feb 20 Update:  
Successfully constructed tensorflow CNN model.   

The structure is a little bit like LeNet: three conv layers followed by two fc layers. It's very likely to be overfitting during training process, 
thus I added dropout after each layer.  

Using Adam Optimizor, which is super efficient, and L2-norm. The best val_acc is 61.8% after 200 epochs training.  


### Feb 21 Update:
Replace the dropout with Batch-norm, and re-design the Conv Layer (using more neurons at small regions), using new preprocessing method (minus mean and divided by standdv).   

The result was amazing, the val_acc climbed to over 70% (72% at most) after 100 epochs training. But it may lead to disastrous overfitting. Thus I again added dropout after fc layer. Then the network started to converge.  


### Feb 22 Update:
It's a little bit confusing that my test result is extremly poor on kaggle test set (8% for the first time). I was thinking about using different preprocessing method like white noise, but it still doesn't work. (13% for the second time) So I double check the training data, the reason why it happening is because I used the wrong dataset! I used the Tensorflow-tutorial version of dataset with totally different labels. 

I reloaded the data and retrained the networks, amazing things happened again. Perhaps the quality of this dataset is better than the tutorial version, the val_acc converged to over 85%.  


