import os
import sys


import tensorflow as tf
from google.protobuf import text_format as pbtf




class ModelSaver(object):

    def save(self, classifier, step=None):
        raise NotImplementedError()




class ModelLoader(object):
    
    def load(self, classifier, step):
        raise NotImplementedError()

    def load_from_graph_def(self, session, step):
        raise NotImplementedError()




class TFSaver(ModelSaver):

    def __init__(self, output_path, classifier):
        self.output_path = output_path
        self.classifier = classifier

        var_name_list = [var.name.split(':')[0] for var in classifier.model_vars]
        self.sub_graph_def = tf.graph_util.extract_sub_graph(
            tf.get_default_graph().as_graph_def(add_shapes=True), [self.classifier.ops.classify_raw.name.split(':')[0]] + var_name_list
            )

        tf.train.write_graph(self.sub_graph_def, os.path.dirname(output_path), 'graph.pbtxt')
        self.saver = tf.train.Saver(classifier.model_vars)

    def save(self, classifier, step):
        print("write checkpoint to dir : " + self.output_path + '-%d'%step)
        self.saver.save(self.classifier.session, self.output_path, global_step=step, write_meta_graph=False, write_state=False)



class TFLoader(ModelLoader):

    def __init__(self, output_path, output_name='checkpoint'):
        self.output_path = output_path
        self.output_name = output_name

    def load_from_graph(self, sess, step):
        # load graph def 
        with tf.gfile.FastGFile(os.path.join(self.output_path, "graph.pbtxt"),'r') as f:
            graph_def = tf.GraphDef()
            pbtf.Parse(f.read(), graph_def)
        tf.import_graph_def(graph_def, name='')
        
        # workaround to get all tensors in graph
        var_name_list = [op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name.startswith('Variable')]
        var_list = [tf.get_default_graph().get_tensor_by_name(var_name+':0') for var_name in var_name_list]

        # load checkpoint-<step>.data
        saver = tf.train.Saver(var_list)
        saver.restore(sess, os.path.join(self.output_path, "%s-%d"%(self.output_name,step or 0)))




class PBFileSaver(ModelSaver):
    '''
        deprecated
    '''

    def __init__(self, output_path, classifier):
        self.output_path = output_path  
        self.classifier = classifier
        self.output_node_names = [self.classifier.ops.classify_raw.name.split(':')[0]]

        self.sub_graph_def = tf.graph_util.extract_sub_graph(tf.get_default_graph().as_graph_def(add_shapes=True), [self.classifier.ops.classify_raw.name.split(':')[0]])

    def save(self, classifier, step):
        constant_graph = tf.graph_util.convert_variables_to_constants(classifier.session, self.sub_graph_def, self.output_node_names)
        with tf.gfile.FastGFile(self.output_path + 'checkpoint-%d.pb'%(step or 0), mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    def load(self, classifier, step):
        raise NotImplementedError()



