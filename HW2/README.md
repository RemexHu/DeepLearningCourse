# HW2

### Feb 20 Update:  
Successfully constructed tensorflow CNN model.   

The structure is a little bit like LeNet: three conv layers followed by two fc layers. It's very likely to be overfitting during training process, 
thus I added dropout after each layer.  

Using Adam Optimizor, which is super efficient, and L2-norm. The best val_acc is 61.8% after 200 epochs training.  


