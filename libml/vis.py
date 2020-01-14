import os
import sys
import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from absl import flags

import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn
import csv
from yattag import Doc
from yattag import indent
import traceback
import contextlib
import cv2
import base64

from scipy.spatial.distance import cdist
from matplotlib import cm

FLAGS = flags.FLAGS


###############################################



class Plotter(object):
	'''
		class Plotter
		将训练过程中的数值化输出保存成html以及csv文件进行记录

		用法：
		1. 初始化：
			plotter = Plotter()

		2. 记录：
			plotter.scalar('loss1', step=10, value=0.1)

		3. 输出到html文件：
			plotter.to_html_report('./experiment/plt')
		
		4. 输出所有数据分别到单独的csv文件中：
			plotter.to_csv('./experiment/plt')
	'''
	def __init__(self):
		self._scalar_data_frame_dict = {}
		self._dist_data_frame_dict = {}

	def _check_dir(self, path):
		p = os.path.dirname(path)
		if not os.path.exists(p):
			os.mkdir(p)
		return p

	def scalar(self, name, step, value, epoch=None):
		if isinstance(value, dict):
			data = value.copy()
			data.update({
				'step' : step
			})
		else:
			data = {
				'step' : step,
				name : value,
			}
		
		if epoch is not None:
			data['epoch'] = epoch
				
		df = pd.DataFrame(data, index=[0])

		if name not in self._scalar_data_frame_dict:
			self._scalar_data_frame_dict[name] = df
		else:
			self._scalar_data_frame_dict[name] = self._scalar_data_frame_dict[name].append(df, ignore_index=True)

	def dist(self, name, step, mean, var, epoch=None):
		if epoch is not None:
			df = pd.DataFrame({'epoch' : epoch, 'step' : step, name+'_mean' : mean, name+'_var' : var }, index=[0])
		else:
			df = pd.DataFrame({'step' : step, name+'_mean' : mean, name+'_var' : var, }, index=[0])

		if name not in self._dist_data_frame_dict:
			self._dist_data_frame_dict[name] = df
		else:
			self._dist_data_frame_dict[name] = self._dist_data_frame_dict[name].append(df, ignore_index=True)


	def dist2(self, name, step, value_list, epoch=None):
		mean = np.mean(value_list)
		var = np.var(value_list)
		self.dist(name, step, mean, var, epoch=epoch)


	def to_csv(self, output_dir):
		" 将记录保存到多个csv文件里面，csv文件放在output_dir下面。"
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		for name, data_frame in self._scalar_data_frame_dict.items():
			csv_filepath = os.path.join(output_dir, 'scalar_'+name+'.csv')
			data_frame.to_csv(csv_filepath, index=False)
		for name, data_frame in self._dist_data_frame_dict.items():
			csv_filepath = os.path.join(output_dir, 'dist_'+name+'.csv')
			data_frame.to_csv(csv_filepath, index=False)

	def from_csv(self, output_dir):
		" 从output_dir下面的csv文件里面读取并恢复记录 "
		csv_name_list = [fn.split('.')[0] for fn in os.listdir(output_dir) if fn.endswith('csv')]
		for name in csv_name_list:
			if name.startswith('scalar_'):
				in_csv = pd.read_csv(os.path.join(output_dir, name+'.csv'))
				self._scalar_data_frame_dict[name[len('scalar_'):]] = in_csv
			elif name.startswith('dist_'):
				self._dist_data_frame_dict[name[len('dist_'):]] = pd.read_csv(os.path.join(output_dir, name+'.csv'))


	def write_svg_all(self, output_dir):
		" 将所有记录绘制成svg图片 "
		for ind, (name, data_frame) in enumerate(self._scalar_data_frame_dict.items()):
			try:
				output_svg_filepath = os.path.join(output_dir, name+'.svg')
				plt.figure()
				plt.clf()
				headers = [hd for hd in data_frame.columns if hd not in ['step', 'epoch']]
				if len(headers) == 1:
					plt.plot(data_frame['step'], data_frame[name])
				else:
					for hd in headers:
						plt.plot(data_frame['step'], data_frame[hd])
					plt.legend(headers)
				plt.tight_layout()
				plt.savefig(output_svg_filepath)
				plt.close()
			except Exception as e:
				print('draw %s failed : '%name)
				print(data_frame)
				traceback.print_exc()


		for ind, (name, data_frame) in enumerate(self._dist_data_frame_dict.items()):
			output_svg_filepath = os.path.join(output_dir, name+'.svg')
			plt.figure()
			plt.clf()
			plt.errorbar(data_frame['step'], data_frame[name+'_mean'], yerr=data_frame[name+'_var'])
			plt.tight_layout()
			plt.savefig(output_svg_filepath)
			plt.close()


	def to_html_report(self, output_filepath):
		" 将所有记录整理成一个html报告 "
		self.write_svg_all(self._check_dir(output_filepath))
		doc, tag, text = Doc().tagtext()
		with open(output_filepath, 'w') as outfile:
			with tag('html'):
				with tag('body'):

					with tag('h3'):
						text('1. scalars')

					for ind, (name, data_frame) in enumerate(self._scalar_data_frame_dict.items()):
						with tag('div', style='display:inline-block'):
							with tag('h4', style='margin-left:20px'):
								text('(%d). '%(ind+1)+name)
							doc.stag("embed", style="width:800px;padding:5px;margin-left:20px", src=name+'.svg', type="image/svg+xml")

					with tag('h3'):
						text('2. distributions')

					for ind, (name, data_frame) in enumerate(self._dist_data_frame_dict.items()):
						with tag('div', style='display:inline-block'):
							with tag('h4', style='margin-left:20px'):
								text('(%d). '%(ind+1)+name)
							doc.stag("embed", style="width:800px;padding:5px;margin-left:20px", src=name+'.svg', type="image/svg+xml")

			result = indent(doc.getvalue())
			outfile.write(result)




#############################################################################



class RecordInterface(object):
	def start_draw(self):
		raise NotImplementedError()

	def draw_image(self, image, title=''):
		raise NotImplementedError()

	def _check_dir(self, path):
		if not os.path.exists(path):
			os.mkdir(path)
		return path



class RecordData(object):
	def draw(self, draw_interface):
		raise NotImplementedError()




class HtmlRecorder(RecordInterface):
	# def __init__(self, ):
	# 	self.output_path = output_path
	# 	self.filename = filename

	@contextlib.contextmanager
	def start_draw(self, output_path, filename='index.html'):
		self._check_dir(output_path)
		self.doc, self.tag, self.text = Doc().tagtext()
		with open(os.path.join(output_path, filename), 'w') as outfile:
			with self.tag('html'):
				with self.tag('body'):
					self.index = 1
					yield
			result = indent(self.doc.getvalue())
			outfile.write(result)

	def draw_image(self, image, title=''):
			with self.tag('h3'):
				self.text('%d : %s'%(self.index, title))
				self.index += 1
			with self.tag('div', style='display:inline-block'):
				self.doc_image(image)

	def doc_image(self, img, width=500):
		img_buf = cv2.imencode('.jpg', img)[1]
		self.doc.stag("img", style="width:%dpx;padding:5px"%width, src="data:image/png;base64,"+str(base64.b64encode(img_buf))[2:-1])



class RecordDataHelper(RecordData):

	def cv2_imread(self, filepath):
		filein = np.fromfile(filepath, dtype=np.uint8)
		cv_img = cv2.imdecode(filein, cv2.IMREAD_COLOR)
		return cv_img

	def img_vertical_concat(self, images, pad=0, pad_value=255, pad_right=False):
		nb_images = len(images)
		h_list = [i.shape[0] for i in images]
		w_list = [w.shape[1] for w in images]
		if pad_right:
			max_w = np.max(w_list)
			images = [i if i.shape[1] == max_w else np.hstack([i, np.ones([i.shape[0], max_w-i.shape[1]]+list(i.shape[2:]), dtype=i.dtype)*pad_value])
					for i in images]
		else:
			assert np.all(np.equal(w_list, w_list[0]))

		if pad != 0:
			images = [np.vstack([i, np.ones([pad,]+list(i.shape[1:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

		if not isinstance(images, list):
			images = [i for i in images]
		return np.vstack(images)

	def img_horizontal_concat(self, images, pad=0, pad_value=255, pad_bottom=False):
		nb_images = len(images)
		h_list = [i.shape[0] for i in images]
		w_list = [w.shape[1] for w in images]
		if pad_bottom:
			max_h = np.max(h_list)
			images = [i if i.shape[0] == max_h else np.vstack([i, np.ones([max_h-i.shape[0]]+list(i.shape[1:]), dtype=i.dtype)*pad_value])
					for i in images]
		else:
			assert np.all(np.equal(h_list, h_list[0]))

		if pad != 0:
			images = [np.hstack([i, np.ones([i.shape[0], pad,]+list(i.shape[2:]), dtype=i.dtype)*pad_value]) for i in images[:-1]] + [images[-1],]

		if not isinstance(images, list):
			images = [i for i in images]
		return np.hstack(images)
		
	def img_grid(self, images, nb_images_per_row=10, pad=0, pad_value=255):
		ret = []
		while len(images) >= nb_images_per_row:
			ret.append(self.img_horizontal_concat(images[0:nb_images_per_row], pad_bottom=True, pad=pad, pad_value=pad_value))
			images = images[nb_images_per_row:]
		if len(images) != 0:
			ret.append(self.img_horizontal_concat(images, pad_bottom=True, pad=pad, pad_value=pad_value))
		return self.img_vertical_concat(ret, pad_right=True)

	def label_img(self, img, lbl):
		img = np.vstack([img, np.ones([25,]+list(img.shape[1:]), dtype=img.dtype) * 255])
		return cv2.putText(img, str(lbl), (3, img.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)


class FeatureSpaceData(RecordDataHelper):

	def __init__(self, name, feature_list, data_source):
		self.name = name
		self.feature_list = np.array(feature_list, dtype=np.float32)
		self.data_source = data_source

	def to_image(self, select_images=10, nearest_images=10):
		s_indices = np.random.choice(np.arange(len(self.feature_list)), size=select_images, replace=False)
		s_feature_list = self.feature_list[s_indices]
		distances = cdist(s_feature_list, self.feature_list)
		nearest_indices = np.argsort(distances, axis=1)[:, 1:1+nearest_images]

		image_list = []
		for s_ind, n_inds in zip(s_indices, nearest_indices):

			s_img = self.data_source.get_img(s_ind)
			# pad img 
			h = s_img.shape[0]
			pad_img = np.ones([h, 30] + list(s_img.shape[2:]), dtype=s_img.dtype) * 255

			near_img_list = [self.data_source.get_img(ind) for ind in n_inds]
			image_list.append(self.img_horizontal_concat([s_img, pad_img] + near_img_list))

		return self.img_vertical_concat(image_list)

	def draw(self, draw_interface):
		draw_interface.draw_image(title=self.name, image=self.to_image())


class FeatureSpaceDataWLabel(FeatureSpaceData):


	def to_image(self, select_range=None, select_images=10, nearest_images=10):
		if select_range is None:
			s_indices = np.random.choice(np.arange(len(self.feature_list)), size=select_images, replace=False)
		else:
			s_indices = np.random.choice(np.arange(*select_range), size=select_images, replace=False)
		s_feature_list = self.feature_list[s_indices]
		distances = cdist(s_feature_list, self.feature_list)
		nearest_indices = np.argsort(distances, axis=1)[:, 1:1+nearest_images]

		image_list = []
		for s_ind, n_inds in zip(s_indices, nearest_indices):

			s_img = self.data_source.get_img(s_ind)
			s_lbl = self.data_source.get_cls(s_ind)
			s_img = self.label_img(s_img, s_lbl)
			# pad img 
			h = s_img.shape[0]
			pad_img = np.ones([h, 30] + list(s_img.shape[2:]), dtype=s_img.dtype) * 255

			near_img_list = [self.data_source.get_img(ind) for ind in n_inds]
			near_lbl_list = [self.data_source.get_cls(ind) for ind in n_inds]
			near_img_list = [self.label_img(img, lbl) for img, lbl in zip(near_img_list, near_lbl_list)]
			image_list.append(self.img_horizontal_concat([s_img, pad_img] + near_img_list))

		return self.img_vertical_concat(image_list)



class FeatureSpaceData2(FeatureSpaceDataWLabel):

	def __init__(self, name, feature_list, data_source, labeled_size):
		self.name = name
		self.feature_list = np.array(feature_list, dtype=np.float32)
		self.data_source = data_source
		self.labeled_size = labeled_size


	def to_image(self, select_range=None, select_images=10, nearest_images=15):
		if select_range is None:
			s_indices = np.random.choice(np.arange(len(self.feature_list)), size=select_images, replace=False)
		else:
			s_indices = np.random.choice(np.arange(*select_range), size=select_images, replace=False)
		s_feature_list = self.feature_list[s_indices]
		distances = cdist(s_feature_list, self.feature_list)
		nearest_indices = np.argsort(distances, axis=1)[:, 1:1+nearest_images]

		def get_lbl(ind):
			return self.data_source.get_cls(ind) + ('_l' if ind < self.labeled_size else '_u')

		image_list = []
		for s_ind, n_inds in zip(s_indices, nearest_indices):

			s_img = self.data_source.get_img(s_ind)
			s_lbl = get_lbl(s_ind)
			s_img = self.label_img(s_img, s_lbl)
			# pad img 
			h = s_img.shape[0]
			pad_img = np.ones([h, 30] + list(s_img.shape[2:]), dtype=s_img.dtype) * 255

			near_img_list = [self.data_source.get_img(ind) for ind in n_inds]
			near_lbl_list = [get_lbl(ind) for ind in n_inds]
			near_img_list = [self.label_img(img, lbl) for img, lbl in zip(near_img_list, near_lbl_list)]
			image_list.append(self.img_horizontal_concat([s_img, pad_img] + near_img_list))

		return self.img_vertical_concat(image_list)

	def draw(self, draw_interface):
		draw_interface.draw_image(title=self.name + " labeled data", image=self.to_image([0, self.labeled_size]))
		draw_interface.draw_image(title=self.name + " unlabeled data", image=self.to_image([self.labeled_size, len(self.feature_list)]))


class ImageVSTensorData(RecordDataHelper):
	def __init__(self, name, image, data, point_out_ind=None, channel_wise_normalize=False):
		
		assert len(image) == len(data)
		assert len(data.shape) == 4

		if channel_wise_normalize:
			data = self.normalization(data, normalize_axis=(1, 2,))
		else:
			data = self.normalization(data, normalize_axis=(1, 2, 3,))
			

		self.name = name
		self.image = image
		self.data = data
		self.point_out_ind = point_out_ind
		self.cmap = cm.ScalarMappable()
	
	def normalization(self, data, normalize_axis=None):
		channel_max = np.max(data, axis=normalize_axis, keepdims=True)
		channel_min = np.min(data, axis=normalize_axis, keepdims=True)
		data = (data - channel_min) / (channel_max - channel_min + 1e-5)
		return data

	def convert_to_color(self, image):
		ret = self.cmap.to_rgba(image, bytes=True, norm=False)
		return ret

		# img = np.repeat(image[:, :, np.newaxis], 3, axis=2)
		# color1 = np.zeros_like(img, dtype=np.uint8)
		# color1[:, :, 0] = 255
		# color2 = np.zeros_like(img, dtype=np.uint8)
		# color2[:, :, 2] = 255
		# ret = (img * color2 + (1.0 - img) * color1).astype(np.uint8)
		# return ret

	def draw(self, draw_interface, ):
		
		batch_num = len(self.image)
		num_channels = int(self.data.shape[-1])
		num_col = int(np.ceil(np.sqrt(num_channels)))
		num_row = int(np.ceil(float(num_channels) / float(num_col)))

		for i in range(batch_num):

			data = self.data[i]
			grid_imgs = [self.convert_to_color(data[:, :, i]) for i in range(data.shape[2])]
		
			left_image = self.convert_to_color(self.image[i])

			right_image = self.img_grid(grid_imgs, nb_images_per_row=num_col)

			middle_images = []
			if self.point_out_ind is not None:
				for name, value in self.point_out_ind.items():
					img = self.normalization(data[:, :, value[i]])
					img = self.convert_to_color(img)
					img = self.label_img(img, name +':%d'%(value[i]))
					middle_images.append(img)

			out_image = self.img_horizontal_concat([left_image, ] + middle_images + [right_image], pad=20, pad_bottom=True)

			draw_interface.draw_image(title=self.name + ' image %d'%(i+1), image=out_image)


class DatasourceViewer(RecordDataHelper):
	def __init__(self, datasource):
		self.data_source = datasource

	def draw(self, interface):
		class_list = np.unique(self.data_source.labels)
		for class_ind in class_list:
			class_image_indices = np.random.choice(np.where(self.data_source.labels == class_ind)[0], size=30, replace=False)
			class_image_list = [self.data_source.get_img(ind) for ind in class_image_indices]
			interface.draw_image(self.img_grid(class_image_list), title='class%s'%str(class_ind))




def AutoVisDecorator(cls):
	class Wrapper(cls):
		def on_epoch_start(self, epoch_ind, epochs):
			super(cls, self).on_epoch_start(epoch_ind, epochs)

			if epoch_ind % 5 != 0:
				return

			eval_dict = OrderedDict(
					train=self.dataset.labeled_data,
					valid=self.dataset.valid_data,
					test=self.dataset.test_data)

			for subset, data_source in eval_dict.items():

				nb_images = data_source.size
				choose_indices = np.random.choice(np.arange(nb_images), size=10, replace=False)
				labels = np.array([data_source.get_label(ind) for ind in choose_indices])
				images = np.array([data_source.get_img(ind) for ind in choose_indices])

				feats, cam = self.session.run(	
					self.ops.cam_op,
					feed_dict={
						self.ops.x: images.astype(np.float32) / 255.0
					})
				data_list = [
					ImageVSTensorData('class activation map', images, cam, point_out_ind={
						'l' : labels,
						'p' : np.argmax(feats.logits, axis=1),
					}, channel_wise_normalize=False),
				]
				watch_out_list = {
					'cnn13' : ['bn2_3', 'bn1_2'],
					'resnet18' : ['res3b', 'res2b', 'res1b']
				}.get(FLAGS.arch, [])
				data_list += [ImageVSTensorData(l, images, feats[l]) for l in watch_out_list]
				draw_data_list_to_html(data_list, os.path.join(self.train_dir, 'cam_view_'+subset), epoch_ind=int(self.session.run(self.epoch)))

	Wrapper.__name__ = cls.__name__
	Wrapper.__doc__ = cls.__doc__
	
	return Wrapper



###########################################################################


def draw_data_list_to_html(data_list, output_path, epoch_ind=None):
	'''
		将RecordData数据类列表绘制到输出文件中

		data_list ：由RecordData派生出来子类的实例组成
	'''
	recorder = HtmlRecorder()
	with recorder.start_draw(output_path):
		for data in data_list:
			data.draw(recorder)
	if epoch_ind is not None:
		# keep recording the change in each epoch
		shutil.copy(os.path.join(output_path, 'index.html'), os.path.join(output_path, 'index_epoch%d.html'%epoch_ind))


if __name__ == "__main__":
	p = Plotter()

	# p.scalar('loss', 1, 100)
	# p.scalar('loss', 2, 100)
	# p.scalar('loss', 3, 100)
	# p.scalar('loss', 4, 100)
	# p.scalar('loss', 5, 100)
	# p.scalar('loss', 6, 100)
	# p.scalar('loss', 7, 100)
	# p.scalar('loss', 8, 100)
	# p.scalar('loss', 9, 100)
	# p.scalar('loss', 10, 100)
	# p.scalar('loss', 11, 100)
	# p.scalar('loss', 12, 100)

	# p.scalar('loss2', 1, 100)
	# p.scalar('loss2', 2, 100)
	# p.scalar('loss2', 3, 100)
	# p.scalar('loss2', 4, 100)
	# p.scalar('loss2', 5, 100)
	# p.scalar('loss2', 6, 100)
	# p.scalar('loss2', 7, 100)
	# p.scalar('loss2', 8, 100)
	# p.scalar('loss2', 9, 100)
	# p.scalar('loss2', 10, 100)
	# p.scalar('loss2', 11, 100)
	# p.scalar('loss2', 12, 100)

	# p.dist('loss3', 1, 100.0/1.0, 10)
	# p.dist('loss3', 2, 100.0/2.0, 10)
	# p.dist('loss3', 3, 100.0/3.0, 10)
	# p.dist('loss3', 4, 100.0/4.0, 10)
	# p.dist('loss3', 5, 100.0/5.0, 10)
	# p.dist('loss3', 6, 100.0/6.0, 10)
	# p.dist('loss3', 7, 100.0/7.0, 10)
	# p.dist('loss3', 8, 100.0/8.0, 10)
	# p.dist('loss3', 9, 100.0/9.0, 10)
	# p.dist('loss3', 10, 100.0/10.0, 10)
	# p.dist('loss3', 11, 100.0/11.0, 10)
	# p.dist('loss3', 12, 100.0/12.0, 10)

	# p.from_csv('./experiments/main/a/plot_output')
	# # print(p._scalar_data_frame_dict.headers)
	# p.to_html_report('./experiments/main/a/plot_output/output.html')

	# p.to_csv('./test_csv_output2')

