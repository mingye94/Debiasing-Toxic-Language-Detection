import tensorflow as tf
import tqdm
import time

def test_imdb(dataset, model, log_path, idx2word):
    """
    Conventional testing of a classifier.
    """
    avg_env_inv_acc = tf.contrib.eager.metrics.Accuracy("avg_env_inv_acc",
                                                        dtype=tf.float32)
    avg_env_enable_acc = tf.contrib.eager.metrics.Accuracy("avg_env_enable_acc",
                                                           dtype=tf.float32)

    selected_bias = 0
    num_sample = 0
    if log_path is not None:
        fw = open(log_path, 'w')

    for (batch, (inputs, masks, labels, envs)) in enumerate(tqdm.tqdm(dataset)):
        
        rationale, env_inv_logits, env_enable_logits = model(
            inputs, masks, envs)
        tensor_pred_inv = tf.argmax(env_inv_logits, axis=1, output_type=tf.int64)
        tensor_pred_ena = tf.argmax(env_enable_logits, axis=1, output_type=tf.int64)
        tensor_labels = tf.argmax(labels, axis=1, output_type=tf.int64)
        
        avg_env_inv_acc(tensor_pred_inv,
                        tensor_labels)
        avg_env_enable_acc(
            tensor_pred_ena,
            tensor_labels)
        

        # calculate the percentage that the added bias term is highlighted
        selected_bias += tf.reduce_sum(rationale[:, 0, 1])
        num_sample += inputs.get_shape().as_list()[0]
        
        rationale = list(rationale.numpy())
        inputs = list(inputs.numpy())
        if log_path is not None:
            to_write = []
            for idx, (rat, inp) in enumerate(zip(rationale, inputs)):
                
                text = [idx2word[int(x)] for x in inp if x > 0]
                # text = ['o']*300
                rat_text = []
                rat_nums = []
                
                for i, r in enumerate(rat):
                    if i == len(text):
                        break
                    if r[1] == 1:
                        rat_text.append(text[i])
                        rat_nums.append(i)
                
                # fw.write(str(int(tensor_pred_inv[idx])) + '\t' + str(int(tensor_pred_ena[idx])) + '\t' + str(int(tensor_labels[idx])) + '\t' + ' '.join([x for x in text if x != "<pad>"]) + '\t' + ' '.join(rat_text) + '\t' + ' '.join([str(int(x)) for x in rat_nums]) + '\n')
                to_write.append(str(int(tensor_pred_inv[idx])) + '\t' + str(int(tensor_pred_ena[idx])) + '\t' + str(int(tensor_labels[idx])) + '\t' + ' '.join([x for x in text if x != "<pad>"]) + '\t' + ' '.join(rat_text) + '\t' + ' '.join([str(int(x)) for x in rat_nums]) + '\n')
                
            fw.writelines(to_write)
            
        # print(' '.join(["%.4f"%(times[x]-times[x-1]) for x in range(1, len(times))]), flush=True)

    bias_ = selected_bias / float(num_sample)

    print("{:s}{:.4f}, {:s}{:.4f}, {:s}{:.4f}.".format(
        "----> [Eval] env inv acc: ", avg_env_inv_acc.result(),
        "env enable acc: ", avg_env_enable_acc.result(), "bias selection: ",
        bias_),
          flush=True)

    return avg_env_inv_acc.result(), avg_env_enable_acc.result(), bias_
