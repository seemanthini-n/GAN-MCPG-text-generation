from __future__ import print_function
import numpy as np
import random

"""Performs model training
	sess: tensorflow session
    trained_model: model to be trained
    iteration_count: number of iteration steps/count
    supervised_proportion: iteration proportion for supervisor
    generator_steps: count of generator training steps per iteration
    discriminator_steps: count of discriminator training steps per iteration
    next_sequence: returns predicted/sequence of next timestep
    check_sequence: validated the generated sequence with T/F
    words: indices mapping to word array  
    generator_proportion: ratio of discriminator training on generated data

    """
def train_epoch(sess, trained_model, iteration_count,
                supervised_proportion, generator_steps, discriminator_steps,
                next_sequence, check_sequence=None,
                words=None,
                generator_proportion=0.5):
    
    supervised_gen_loss = [0]  
    unsupervised_gen_loss = [0]  
    discriminator_loss = [0]
    feedback_rewards = [[0] * trained_model.seq_len]
    supervised_true_generation = [0]
    unsupervised_true_generation = [0]
    supervised_gen_output = None
    unsupervised_gen_output = None
    print('executing %d iterations with %d generator steps and %d discriminator steps' % (iteration_count, generator_steps, discriminator_steps))
    print('out of the gen steps, %.2f will be supervised' % supervised_proportion)
    for it in range(iteration_count):
        for _ in range(generator_steps):
            if random.random() < supervised_proportion:
                text_sequence = next_sequence()
                _, generator_loss, generator_prediction = trained_model.pretraining_step(sess, text_sequence)
                supervised_gen_loss.append(generator_loss)

                supervised_gen_output = np.argmax(generator_prediction, axis=1)
                if check_sequence is not None:
                    supervised_true_generation.append(
                        check_sequence(supervised_gen_output))
            else:
                _, _, generator_loss, expected_reward, unsupervised_gen_output = \
                    trained_model.training_generator_step(sess)
                feedback_rewards.append(expected_reward)
                unsupervised_gen_loss.append(generator_loss)

                if check_sequence is not None:
                    unsupervised_true_generation.append(
                        check_sequence(unsupervised_gen_output))

        for _ in range(discriminator_steps):
            if random.random() < generator_proportion:
                text_sequence = next_sequence()
                _, dis_loss = trained_model.training_dis_authentic_step(sess, text_sequence)
            else:
                _, dis_loss = trained_model.training_dis_generator_step(sess)
            discriminator_loss.append(dis_loss)

    print('data per epoch:')
    dis_loss_list=None
    supervised_gen_loss_list=None
    unsupervised_gen_loss_list=None
    supervised_gen_text=None
    unsupervised_gen_text=None
    print('discriminator loss:', np.mean(discriminator_loss))
    dis_loss_list=np.mean(discriminator_loss)
    
    print('generator loss (supervised: {0}, unsupervised: {1})'.format( np.mean(supervised_gen_loss), np.mean(unsupervised_gen_loss)))
    supervised_gen_loss_list=np.mean(supervised_gen_loss)
    unsupervised_gen_loss_list=np.mean(unsupervised_gen_loss)
    
    if check_sequence is not None:
        print('true generations (supervised:{0}, unsupervised: {1})'.format( np.mean(supervised_true_generation), np.mean(unsupervised_true_generation)))
    print('sampled generations (supervised)\n')
    #storing for evaluation
    supervised_gen_text=''.join([words[x] if words else x for x in supervised_gen_output]) if supervised_gen_output is not None else None
    print(''.join([words[x] if words else x for x in supervised_gen_output]) if supervised_gen_output is not None else None,)
    #storing for evaluation
    unsupervised_gen_text=''.join([words[x] if words else x for x in unsupervised_gen_output]) if unsupervised_gen_output is not None else None
    print('sampled generations (unsupervised)\n')
    print(''.join([words[x] if words else x for x in unsupervised_gen_output]) if unsupervised_gen_output is not None else None)
    #print('expected rewards:', np.mean(feedback_rewards, axis=0))
    return dis_loss_list,supervised_gen_loss_list,unsupervised_gen_loss_list,supervised_gen_text,unsupervised_gen_text