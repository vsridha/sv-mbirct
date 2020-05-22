import time

from utils_cnn import *

import os.path

from glob import glob

ckpt_key_type='epoch' #or 'iter'
NumDigitsEpoch=3
VerboseFlag=0

def dncnn(input, is_training=True, output_channels=1):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            # tf.layers.conv2d same as tf.nn.conv2d in terms of functionality.
            # Typical order: conv2d --> BatchNorm --> ReLu --> (optional pooling/upsampling)
            # Note : BatchNorm layer has an input is_training. BatchNorm parameters mu, sigma are learnt during training but fixed while testing. 
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output


class denoiser(object):
    def __init__(self, sess, input_c_dim=1, sigma=25, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.sigma = sigma
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image') #4D - Batch-size, Ny, Nx, Nchannels=1.
        self.is_training  = tf.placeholder(tf.bool, name='is_training')
        self.is_add_noise = tf.placeholder(tf.float32, name='is_add_noise')  # this is actually boolean, but for syntax purposes we represent it as a float
        self.X = self.Y_ + (tf.fill(tf.shape(self.Y_), self.is_add_noise) * tf.random_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0))   # noisy images
        # self.X = self.Y_ + tf.truncated_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        self.Y = dncnn(self.X, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, epoch_num, test_data, sample_files, summary_merged, summary_writer, img_format='png', lower_bound=None, upper_bound=None, is_add_noise=True):
        # assert test_data value range is 0-255 in case of img_format='png' or 'lower_bound'-'upper_bound' in case of img_format='bin'
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in range(len(test_data)):
            
            if(img_format=='png'):
                vh = 255.0
                clean_image = test_data[idx].astype(np.float32) / 255.0
            else:
                vl, vh = lower_bound, upper_bound
                clean_image = (test_data[idx].astype(np.float32)-vl)/(vh-vl)
                clean_image = np.minimum(np.maximum(clean_image, 0), 1)  #clip

            # Denoise    
            output_clean_image, noisy_image, psnr_summary = self.sess.run([self.Y, self.X, summary_merged], 
                                                                          feed_dict={self.Y_: clean_image, self.is_training: False, self.is_add_noise: float(is_add_noise)})
            summary_writer.add_summary(psnr_summary, epoch_num) #iter_num --> epoch_num
            
            # VS: Re-scale data back
            if(img_format=='png'):
                groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
                noisyimage  = np.clip(255 * noisy_image, 0, 255).astype('uint8')
                outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            else:
                groundtruth = test_data[idx].astype('float32')
                noisyimage  = vl+(vh-vl)*np.minimum(np.maximum(noisy_image, 0), 1)
                outputimage = vl+(vh-vl)*np.minimum(np.maximum(output_clean_image, 0), 1)  

            # calculate PSNR
            # psnr = cal_psnr(groundtruth, outputimage)
            psnr=cal_psnr_new(groundtruth, outputimage, vh) #using the new (correct) definition

            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            
            # Save images
            fname = '%s_ep%0*d' % (sample_files[idx], NumDigitsEpoch, epoch_num) #iter_num -->epoch_num
            if(img_format=='png'):
                save_images_png(fname, groundtruth, noisyimage, outputimage, psnr=psnr) 
            else:
                save_images_bin(fname, noisyimage, outputimage, groundtruth, psnr=psnr)

        avg_psnr = psnr_sum / len(test_data)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

 
    def train(self, data, eval_data, batch_size, ckpt_dir, epoch, lr, sample_files, eval_every_epoch=1, img_format='png', lower_bound=None, upper_bound=None):
        
        # ---VS: The patches (originally in range 0 to 255) have already been normalized to 0-1 in function denoiser_train()---
        # ---VS: Validation data in range 0 to 255---
        # ---VS: We assume that for training, the training patches and validation images are clean (we simulate noisy image from clean data), so add_noise=0 for evaluate()---

        # assert data range is between 0 and 1
        numBatch = int(data.shape[0] / batch_size)

        print('Training: Num-batches=%d, Batch-size=%d\n'%(numBatch, batch_size))

        # load pretrained model
        load_model_status, ckpt_key = self.load(ckpt_dir)
       
        # To load from a specific check-point
        #load_model_status, ckpt_key = self.load(ckpt_dir, ckpt_state='specific', ckpt_key_in=<epoch_num>)

        if load_model_status:
            if(ckpt_key_type=='epoch'):
                iter_num    = ckpt_key*numBatch
                start_epoch = ckpt_key
                start_step  = 0
            else:
                iter_num = ckpt_key
                start_epoch = ckpt_key // numBatch
                start_step  = ckpt_key % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        
        # make summary (for Tensorboard visualization)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr) #VS: this does not need to be changed since patch value always in 0-1 range

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()

        self.evaluate(start_epoch, eval_data, sample_files=sample_files, summary_merged=summary_psnr, summary_writer=writer, 
                      img_format=img_format, lower_bound=lower_bound, upper_bound=upper_bound, is_add_noise=True)  

        # This way of stating a for-loop works (since range is computed once at start) though not advisable
        for epoch in range(start_epoch, epoch):
            np.random.shuffle(data)
            loss_sum=0
            for batch_id in range(start_step, numBatch):
                batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                # batch_images = batch_images.astype(np.float32) / 255.0 # normalize the data to 0-1
                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                                                 feed_dict={self.Y_: batch_images, self.lr: lr[epoch],
                                                            self.is_training: True, self.is_add_noise: float(True)})
                if(VerboseFlag):
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                     % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                loss_sum += loss 
                iter_num += 1
                writer.add_summary(summary, iter_num)

            if np.mod(epoch + 1, eval_every_epoch) == 0:
                print("Loss (averaged over all %d batches) = %.6f \n" % (numBatch,(loss_sum/numBatch)) )
                self.evaluate(epoch+1, eval_data, sample_files=sample_files, summary_merged=summary_psnr, summary_writer=writer, 
                              img_format=img_format, lower_bound=lower_bound, upper_bound=upper_bound, is_add_noise=True)  
                if(ckpt_key_type=='epoch'):
                    self.save(epoch+1, ckpt_dir)
                else:
                    self.save(iter_num, ckpt_dir)

        print("[*] Finish training.")


    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        # how does writing (saving) checkpoints using tf.train.Saver() work ? 
        # saving checkpoint at a given time / state writes 3 files : .data, .meta, .index 
        # .data - parameter values, .meta - computational graph., .index - timestamp / state
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num) #the session has a sess.graph that contains the graph

    def load(self, checkpoint_dir, ckpt_state='latest', ckpt_key_in=0):
        # restoring checkpoint is trickier than save. 
        # when storing TF just loads the data (parameters) from the .data checkpoint file. But how about the graph strcuture ?
        # We have two options : 1) Use tf.train.import_meta_graph('<filename>.meta') to read in the graph structure, or, 
        #                       2) Re-define the computational graph 
        # In this code, we use option 2) (the graph structure is defined by the denoiser() object constructor __init__. Object declared each time we start a session)
        
        # Avoiding use of function tf.train.latest_checkpoint since it gives issues once you move ckpt files to new directory
        
        ckpt_index_files = glob((checkpoint_dir+'/'+'*.index'))
        ckpt_key=0

        if(ckpt_state=='latest'):
            for i in range(len(ckpt_index_files)):
                fname   = ckpt_index_files[i].split('/')[-1]
                fname_no_ext  = fname.split('.')[0]
                key = int(fname_no_ext.split('-')[-1])
                if(ckpt_key < key):
                    ckpt_key=key
                    ckpt_file=fname_no_ext
        else:
            fname   = ckpt_index_files[0].split('/')[-1]
            fname_no_ext  = fname.split('.')[0]   
            ckpt_key = ckpt_key_in
            ckpt_file = fname_no_ext+'-'+ckpt_key

        print("[*] Reading checkpoint from epoch %d..." % (ckpt_key))
        saver = tf.train.Saver()
        full_path=checkpoint_dir+'/'+ckpt_file
        saver.restore(self.sess, full_path)
        return True, ckpt_key


    # ----New version ------------    
    def test(self, test_data, ckpt_dir, save_files, img_format='png', ckpt_state='latest', is_add_noise=True, ckpt_epoch_num=None, lower_bound=None, upper_bound=None):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(save_files) != 0, 'No testing data!'
        
        # load pretrained model
        load_model_status, ckpt_key = self.load(ckpt_dir, ckpt_state=ckpt_state, ckpt_key_in=ckpt_epoch_num)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")
        
        #Initialize psnr
        if(is_add_noise):
            psnr_sum = 0
            print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        
        for idx in range(len(save_files)):      
            # Scale image
            if(img_format=='png'):
                vh = 255.0
                input_image = test_data[idx].astype(np.float32) / 255.0
            else:
                vl, vh = lower_bound[idx], upper_bound[idx]
                input_image = (test_data[idx].astype(np.float32)-vl)/(vh-vl)
                input_image = np.minimum(np.maximum(input_image, 0), 1)  #clip

            # Denoise: if add_noise=1 then input image is clean and noise is added to input to generate noisy image X which is sent through denoiser    
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X], 
                                                            feed_dict={self.Y_: input_image, self.is_training: False, self.is_add_noise: float(is_add_noise)})
            
            if(img_format=='png'):
                noisyimage  = np.clip(255 * noisy_image, 0, 255).astype('uint8')
                outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            else:
                noisyimage  = vl+(vh-vl)*np.minimum(np.maximum(noisy_image, 0), 1)
                outputimage = vl+(vh-vl)*np.minimum(np.maximum(output_clean_image, 0), 1)

            if(is_add_noise):  
                groundtruth = test_data[idx].astype('float32')
            else:
                groundtruth = None

            # calculate PSNR
            if(is_add_noise):
                psnr = cal_psnr_new(groundtruth, outputimage, vh)
                print("img%d PSNR: %.2f" % (idx, psnr))
                psnr_sum += psnr
            else:
                psnr=None
            
            if(img_format=='png'):
                save_images_png(save_files[idx], groundtruth, noisyimage, outputimage, psnr=psnr)
            else:
                save_images_bin(save_files[idx], noisyimage, outputimage, groundtruth, psnr=psnr)
        
        if(is_add_noise):
            avg_psnr = psnr_sum / len(save_files)
            print("--- Average PSNR %.2f ---" % avg_psnr)    

