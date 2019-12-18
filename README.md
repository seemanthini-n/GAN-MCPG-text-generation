# GAN-Monte carlo policy gradient-text generation
Neural text generation model using generative adversarial network and monte carlo policy gradient. 
The generator and discriminator are alternatively made to optimise their objectives.
Monte carlo policy gradient helps the generator in selecting the most appropriate discrete valued character at each time step with unsupervised training. Switch rate controls the switch between supervised and unsupervised training. GRU is used as both generator and discriminator. The implementation has been done in tensorflow. 
