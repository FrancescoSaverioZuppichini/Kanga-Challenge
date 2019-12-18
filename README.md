# Smash Bros Melee

## Problem
Identify different features of smash bros melee in real-time from this [clip](https://www.youtube.com/watch?v=bj7IX18ccdY) from Evo 2014. 

![alt](https://github.com/FrancescoSaverioZuppichini/Kanga-Challenge/blob/develop/doc/images/830.jpg?raw=true)

- [ ] what stage was picked 
- [ ] when the game starts 
- [ ] player names 
- [ ] how many stocks the player has 
- [ ] how much percent the player is 
- [ ] in-game timer 
- [ ] draw bounding around the players 
- [ ] which character is been played

Just an example of what is detected in the video. 

I forgot to convert the image to RGB in the gif :)

![alt](https://github.com/FrancescoSaverioZuppichini/Kanga-Challenge/blob/develop/doc/images/sample.gif?raw=true)

## Solution 
I divided the problem into two main stages:
- find the location of stuff on the screen (e.g. the timer on the top)
- get information (text etc) from the regions identified by the first step 

### Find stuff 
So I trained tiny [yolo v3](https://arxiv.org/abs/1804.02767) using this repo () and around 250 manually labeled images. Givin the first image showed at the beginning as input this is the output.

![alt](https://github.com/FrancescoSaverioZuppichini/Kanga-Challenge/blob/develop/doc/images/yolo_det_830.png?raw=true)

The classes are: 
- player
- time
- stocks
- damage

We are not classifying player1 vs player2. We want to be as general as possible.


Not bad! Some details:
- the batch number was changed to 64 so I can use almost all my GTX 1080ti memory 
- learning rate double to 0.005
- the repo code was heavily modified (no OOP and weird stuff)
- no transfer learning. The domain is too different.
- IOU of 1.4 and F1 of 0.91 (gg)

What can be improved? 
- [ ] we are not using the time information. We should try something like [ROLO](https://arxiv.org/pdf/1607.05781.pdf)

### Identify stuff
In this section, I explain in detail how we classified (or the idea) each class. 

All the predictions are made using the first image as input.

#### When the game started. 
The game is started as soon as we have a prediction from our model. This was easy 😎.

#### Time 
Using Tesseract (under the right configuration), we can get the text out of it. For example, by feeding the `time` box to tesseract we get `073902`. This is correct, but sometimes the output is not so accurate. 

How we can improve it? Well, we can define a heuristic to decide what time is it that takes the time from the game and the time from the machine and decide which one is more correct.

#### Stocks# Smash Bros Melee

## Problem
Identify different features of smash bros melee in real time from this [clip](https://www.youtube.com/watch?v=bj7IX18ccdY) from Evo 2014. 

![alt](https://github.com/FrancescoSaverioZuppichini/Kanga-Challenge/blob/develop/doc/images/830.jpg?raw=true)

- [ ] what stage was picked 
- [ ] when the game starts 
- [ ] player names 
- [ ] how many stocks the player has 
- [ ] how much percent the player is 
- [ ] in-game timer 
- [ ] draw bounding around the players  
- [ ] which character is been played

Just an example of what it is detected in the video. 

I forgot to convert the image to RGB in the gif :)

![alt](https://github.com/FrancescoSaverioZuppichini/Kanga-Challenge/blob/develop/doc/images/sample.gif?raw=true)

## Solution 
I divided the problem in two main stages:
- find the location of stuff on the screen (e.g. the timer on the top)
- get information (text etc) from the regions identified by the first step 

### Find stuff 
So I trained tiny [yolo v3](https://arxiv.org/abs/1804.02767) using this repo () and around 250 manually labeled images. Givin the first image showed at the beginning as input this is the output.

![alt](https://github.com/FrancescoSaverioZuppichini/Kanga-Challenge/blob/develop/doc/images/yolo_det_830.png?raw=true)

The classes are: 
- player
- time
- stocks
- damage

We are not classifing player1 vs player2. We want to be as general as possibile.


Not bad! Some details:
- batch was changed to 64 so I can use almost all my GTX 1080ti memory 
- learning rate double to 0.005
- the repo code was heavily modified (no OOP and weird stuff)
- no trasfer learning. The domain is too different.
- IOU of 1.4 and F1 of 0.91 (gg)

What can be improved. 
- [ ] we are not using the time information. We should try something like [ROLO](https://arxiv.org/pdf/1607.05781.pdf)

### Identify stuff
In this section I explain in detail how we classified (or the idea) each class. 

All the predictions are made using the first image as input. It follows a snapshot of the prediction.

```
,x,y,x2,y2,conf,foo,cls,value
0,21.0,348.0,281.0,522.0,0.9764809,0.99986935,0.0,
1,627.0,367.0,932.0,606.0,0.94360924,0.99997973,0.0,
3,361.0,56.0,650.0,126.0,0.8946889,0.99969137,1.0,073902
2,251.0,549.0,419.0,594.0,0.9356816,0.9999809,2.0,4.0
5,32.0,552.0,156.0,596.0,0.8133566,0.99996555,2.0,3.0
4,81.0,621.0,184.0,687.0,0.89411485,0.99998295,3.0,209
6,347.0,620.0,416.0,691.0,0.44382927,0.9999068,3.0,0
```

The cls value is: 0 = player, 1 = time, 2 = stocks and 3 = damage

#### When the game started. 
The game is started as soon as we have a prediction from our model. This was easy 😎.

#### Time 
Using Tesseract (under the right configuration), we can get the text out of it. For example, by feeding the `time` box to tesseract we get `073902`. This is correct, but sometimes the output is not so accurate.  

How we can improve it? Well we can define an heuristic to decide what time is it that takes the time from the game and the time from the machine and decide which one is more correct.

#### Stocks

I spent a lot of time thinking about a good solution to count stocks and then I had an illumination. If we take a look at the one of the stock box we can notive that each icon is a perfect square (this is true for **all the icons**). 

Then, if we have four icons, it means that the base of the rectangle must be at least 4 times its height. So the number is just `b // h`. The algorithm correctly identified the number of stocks to be `3` and `4`.

**So far we have no way to know which player is linked to which stock**

#### Players 

To identify the player (e.g. pikaciu) we should train a very fast model on a dataset that containes a few images of each character in game. Then, we crop out the image where the player is and we get the prediction. 

Once we know which player is been player, we can link it to the correct stock. 

I had no time to generate such dataset, it is not difficult but time consuming. The best way to do it is to play the game wich each character, crop 3/4 image of each character and remove the backgroun with photoshop or something like that. 

Then I can train my network to classify the images and add random background to improve its robustness.

#### Stocks

I did the same thing that I did with the time, somethings tesseract predictios `O` instead of `0` so I just replace them. Sometimes the model is not able to find the right damage. This can be fixed by, if we don't find one, add the closest bounding box from the last frame to the current prediction 

I spent a lot of time thinking about a good solution to count stocks and then I had an illumination. If we take a look at one of the stock box we can notice that each icon is a perfect square (this is true for **all the icons**). 

Then, if we have four icons, it means that the base of the rectangle must be at least 4 times its height. So the number is just `b // h`. The algorithm correctly identified the number of stocks to be `3` and `4`.

**So far we have no way to know which player is linked to which stock**

#### Players 

To identify the player (e.g. Pikachu) we should train a very fast model on a dataset that contains a few images of each character in-game. Then, we crop out the image where the player is and we get the prediction. 

Once we know which player is been player, we can link it to the correct stock. 

I had no time to generate such a dataset, it is not difficult but time-consuming. The best way to do it is to play the game wich each character, crop 3/4 image of each character and remove the background with photoshop or something like that. 

Then I can train my network to classify the images and add random background to improve its robustness.

#### Stocks

I did the same thing that I did with the time, somethings tesseract predictions `O` instead of `0` so I just replace them. Sometimes the model is not able to find the right damage. This can be fixed by, if we don't find one, add the closest bounding box from the last frame to the current prediction. In our example we detect `209` and `0` so the first prediction is wrong.


#### TODO 

- [ ] code API 