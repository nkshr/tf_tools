import numpy as np
import os
import csv
import sys
import random
import math

class eval_info:
    def init(self, labeled_images_file, labels_file, num_images):
        with open(labels_file, 'r') as f:
            labels = [line.replace('\n', '')  for line in f.readlines()];

        self.cinfo_list = []
        self.top1_rate = 0
        self.top5_rate = 0
        
        class_id = 0
        for label in labels:
            cinfo = class_info()
            cinfo.label = label
            cinfo.class_id = class_id
            self.cinfo_list.append(cinfo)
            class_id += 1
            
        with open(labeled_images_file, 'r') as f:
            for line in f.readlines():                                    
                iinfo = image_info()
                
                toks = line.split(sep=" ", maxsplit=1)
                idx = int(toks[1].replace('\n', ''))
                iinfo.name = toks[0]

                cinfo = self.cinfo_list[idx]
                if len(cinfo.iinfo_list) < num_images or num_images < 0:
                    cinfo.iinfo_list.append(iinfo)
                #self.__cinfo_list[idx].images.append(image)

            for cinfo in self.cinfo_list:
                if len(cinfo.iinfo_list) != num_images and num_images >= 0:
                    print('class', cinfo.class_id, 'has too less images({}).'.format(len(cinfo.iinfo_list)))

    def __init__(self):
        self.cinfo_list = []
        self.top1_rate = 0
        self.top5_rate = 0
                    
    def get_class_count(self):
        return len(self.cinfo_list)

    def get_class_info(self, class_id):
        for cinfo in self.cinfo_list:
            if cinfo.class_id == class_id:
                return cinfo

    def calc_topk_rate(self, k):
        num_topk = 0
        num_images = 0
        for cinfo in self.cinfo_list:
            for iinfo in cinfo.iinfo_list:
                if not iinfo.rank < 0:
                    num_images += 1
                    if iinfo.rank < k and iinfo.rank >= 0:
                        num_topk += 1

        return num_topk / num_images
    
    def calc_top5_rate(self):
        # num_correct_preds = 0
        # num_images = 0
        # for cinfo in self.cinfo_list:
        #     for iinfo in cinfo.iinfo_list:
        #         num_images += 1
        #         if iinfo.rank < 5:
        #             num_correct_preds += 1

        # if  num_images:
        #     self.top5_rate = num_correct_preds / num_images
        # else:
        #     self.top5_rate = 0
        sum_top5_rate = 0
        num_valid_classes = 0
        for cinfo in self.cinfo_list:
            if not cinfo.top5_rate < 0:
                num_valid_classes += 1
                sum_top5_rate += cinfo.top5_rate

        if num_valid_classes:
            self.top5_rate = sum_top5_rate / num_valid_classes
        else:
            self.top5_rate = -1
            
    def calc_top1_rate(self):        
        # num_correct_preds = 0
        # num_images = 0
        
        # for cinfo in self.cinfo_list:
        #     num_images += len(cinfo.iinfo_list)
        #     for iinfo in cinfo.iinfo_list:
        #         if iinfo.rank == 0:
        #             num_correct_preds += 1
        # if num_images:
        #     self.top1_rate = num_correct_preds / num_images
        # else:
        #     self.top1_rate = 0
        sum_top1_rate = 0
        num_valid_classes = 0
        for cinfo in self.cinfo_list:
            if not cinfo.top1_rate < 0:
                num_valid_classes += 1
                sum_top1_rate += cinfo.top1_rate

        if num_valid_classes:
            self.top1_rate = sum_top1_rate / num_valid_classes
        else:
            self.top1_rate = -1
            
    def __iter__(self):
        self.cinfo_list_idx = 0
        return self        

    def __next__(self):
        if self.cinfo_list_idx == len(self.cinfo_list):
            raise StopIteration
        self.cinfo_list_idx += 1
        return self.cinfo_list[self.cinfo_list_idx-1]

    def sort_by_top1_rate(self):
        top1_rate_arr = np.array([cinfo.top1_rate for cinfo in self.cinfo_list])
        sorted_indexes = np.argsort(top1_rate_arr)[::-1]
        sorted_cinfo_list = [self.cinfo_list[index] for index in sorted_indexes]
        self.cinfo_list  = sorted_cinfo_list

    def sort_by_class_id(self):
        class_ids = np.array([cinfo.class_id for cinfo in self.cinfo_list])
        sorted_indexes = np.argsort(class_ids)
        sorted_cinfo_list = [self.cinfo_list[index] for index in sorted_indexes]
        self.cinfo_list = sorted_cinfo_list

    def calc_top1_rate_rank(self):
        top1_rate_arr = np.array([cinfo.top1_rate for cinfo in self.cinfo_list])
        sorted_indexes = np.argsort(top1_rate_arr)[::-1]
        for rank in range(len(sorted_indexes)):
            self.cinfo_list[sorted_indexes[rank]].top1_rate_rank = rank

    def calc_top5_rate_rank(self):
        top5_rate_arr = np.array([cinfo.top5_rate for cinfo in self.cinfo_list])
        sorted_indexes = np.argsort(top5_rate_arr)[::-1]
        for rank in range(len(sorted_indexes)):
            self.cinfo_list[sorted_indexes[rank]].top5_rate_rank = rank

    def write_summary(self, fname):
        text = 'top1_rate,{}\n'.format(self.top1_rate)
        text += 'top5_rate,{}\n'.format(self.top5_rate)
        
        text += 'class_id,label,num_images,top1_rate,top5_rate,top1_rate_rank,top5_rate_rank\n'
        for cinfo in self.cinfo_list:
            text += '{},\"{}\",{},{},{},{},{}\n'.format(cinfo.class_id, cinfo.label, len(cinfo.iinfo_list), cinfo.top1_rate, cinfo.top5_rate, cinfo.top1_rate_rank, cinfo.top5_rate_rank)

        with open(fname, 'w') as f:
            f.write(text)
            f.flush()

    def write(self, dir):
        summary = os.path.join(dir, 'summary.csv')
        self.write_summary(summary)

        for cinfo in self:
            detail = os.path.join(dir, 'class{0:04d}.csv'.format(cinfo.class_id))
            cinfo.write(detail)

    def read_summary(self, fname):
        self.top1_rate = 0
        self.top5_rate = 0

        with open(fname, 'r') as f:
            reader  = csv.reader(f)
            row = next(reader)
            self.top1_rate = row[1]

            row = next(reader)
            self.top5_rate = row[1]

    def read(self, dir):
        self.cinfo_list = []
        self.top1_rate = 0
        self.top5_rate = 0
        
        summary = os.path.join(dir, 'summary.csv')
        self.read_summary(summary)
        
        class_csvs = [class_csv for class_csv in os.listdir(dir) if class_csv.startswith('class')]

        for class_csv in class_csvs:
            cinfo = class_info()
            cinfo.read(os.path.join(dir, class_csv))
            self.cinfo_list.append(cinfo)

    def take_statistcs(self):
        for cinfo in self.cinfo_list:
            cinfo.take_statistics()
        
        #self.top1_rate = self.calc_topk_rate(1)
        #self.top5_rate = self.calc_topk_rate(5)
        self.calc_top1_rate()
        self.calc_top5_rate()
        self.calc_top1_rate_rank()
        self.calc_top5_rate_rank()

#######################class_info#######################
class class_info:
    def __init__(self):
        self.iinfo_list = []
        self.label = ''
        self.class_id = ''
        self.top1_rate = -1
        self.top5_rate = -1
        self.top1_rate_rank = -1
        self.top5_rate_rank = -1
        self.evaluated = False
        
    def calc_top1_rate(self):
        if (len(self.iinfo_list)) == 0:
            self.top1_rate = -1
            return

        if iinfo_list[0].rank < -1:
            self.top1_rate = -1
            return 

        num_top1 = 0
        for iinfo in self.iinfo_list:
            if iinfo.rank == 0:
                num_top1 += 1
        self.top1_rate = num_top1 / len(self.iinfo_list)

    def calc_top5_rate(self):
        if (len(self.iinfo_list)) == 0:
            self.top5_rate = -1
            return

        if iinfo_list[0].rank < -1:
            self.top1_rate = -1
            return
        
        num_top5 = 0
        for iinfo in self.iinfo_list:
            if iinfo.rank < 5 and iinfo.rank > 0:
                num_top5 += 1
        self.top5_rate = num_top5 / len(self.iinfo_list)

    def calc_topk_rate(self, k):
        if (len(self.iinfo_list)) == 0:
            return -1

        if self.iinfo_list[0].rank < 0:
            return -1
        
        num_topk = 0
        for iinfo in self.iinfo_list:
            if iinfo.rank < k and iinfo.rank >= 0:
                num_topk += 1

        return num_topk / len(self.iinfo_list)

    def write(self, fname):
        text = 'class_id,{}\n'.format(self.class_id)
        text += 'label,\"{}\"\n'.format(self.label)
        text += 'top1_rate,{}\n'.format(self.top1_rate)
        text += 'top5_rate,{}\n'.format(self.top5_rate)
        text += 'top1_rate_rank,{}\n'.format(self.top1_rate_rank)
        text += 'top5_rate_rank,{}\n'.format(self.top5_rate_rank)
        text += 'image,rank,rank_in_class,prob,1,,2,,3,,4,,5\n'
        
        with open(fname, 'w') as f:
            f.write(text)
            f.flush()

            for iinfo in self.iinfo_list:
                text = '{},{},{},{},'.format(
                    iinfo.name,
                    iinfo.rank,
                    iinfo.rank_in_class,
                    iinfo.prob
                )

                for i in range(5):
                    class_id = iinfo.top5[i]['class_id']
                    prob = iinfo.top5[i]['prob']
                    text += '{},{},'.format(class_id, prob)

                text += '\n'

                f.write(text)
                f.flush()

    def sort_by_prob(self):
        probs = [iinfo.prob for iinfo in self.iinfo_list]
        sorted_indexes = np.argsort(probs)[::-1]
        sorted_iinfo_list = [iinfo_list[index] for index in sorted_indexes]
        self.iinfo_list = sorted_info_list

    def calc_rank_in_class(self):
        probs = []
        for iinfo in self.iinfo_list:
            probs.append(iinfo.prob)

        sorted_indexes = np.argsort(probs)[::-1]
        for rank in range(len(sorted_indexes)):
            self.iinfo_list[sorted_indexes[rank]].rank_in_class = rank

    def read(self, fname):
        iinfo_list = []
        
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            toks = next(reader)
            self.class_id = int(toks[1])

            toks = next(reader)
            self.label = toks[1]

            toks = next(reader)
            self.top1_rate = float(toks[1])

            toks = next(reader)
            self.top5_rate = float(toks[1])

            toks = next(reader)
            self.top1_rate_rank = int(toks[1])

            toks = next(reader)
            self.top5_rate_rank = int(toks[1])
            
            next(reader) #skip header
            for row in reader:
                iinfo = image_info()
                iinfo.name = row[0]
                iinfo.rank = int(row[1])
                iinfo.rank_in_class = int(row[2])
                iinfo.prob = float(row[3])
                for i in range(5):
                    class_id = row[4+i*2]
                    prob = row[5+i*2]
                    iinfo.top5[i]['class_id'] = class_id
                    iinfo.top5[i]['prob'] = prob

                self.iinfo_list.append(iinfo)
                
    def take_statistics(self):
        self.top1_rate = self.calc_topk_rate(1)
        self.top5_rate = self.calc_topk_rate(5)
        self.calc_rank_in_class()

    def calc_top5_mispreds(self):
        mispred_votes = [0 for i in self.get_class_count()]
        mispred_votes[self.class_id] = -1
        
        for iinfo in iinfo_list:
            for rank in range(5):
                class_id = iinfo.top5[rank]['class_id']
                if class_id == self.class_id:
                    break

                point = 5 - class_id
                mispred_votes[class_id] += point

        sorted_indexes = np.argsort(mispred_votes)[::-1]
        self.top5_mispreds = [sorted_indexes[rank] for rank in range(5)]
        
###################image_info##################
class image_info:
    def __init__(self):
        self.name=''
        self.rank = -1
        self.rank_in_class = -1
        self.prob=-1
        self.top5=[{'class_id' : -1, 'prob' : -1} for i in range(5)] #class_id, prob

##################eval_info_comp###############
class eval_info_comp:
    def __init__(self):
        pass

    def read(self, left, right):
        self.left_einfo = eval_info()
        self.left_einfo.read(left)
        self.left_einfo.sort_by_class_id()
        
        self.right_einfo = eval_info()
        self.right_einfo.read(right)
        self.right_einfo.sort_by_class_id()

        if self.left_einfo.get_class_count() != self.right_einfo.get_class_count():
            raise ValueError('{}({}) and {}({}) have same number of classes.'.format(left, left_einfo.get_class_count(), right,right_einfo.get_class_count()))
                
    def take_synthesis(self):
        self.roc_list = [-1 for i in range(self.right_einfo.get_class_count())]

        for class_id in range(self.left_einfo.get_class_count()):
            left_cinfo = self.left_einfo.get_class_info(class_id)
            right_cinfo = self.right_einfo.get_class_info(class_id)
            if left_cinfo.top1_rate < sys.float_info.epsilon or right_cinfo.top1_rate < 0:
                self.roc_list[class_id] = -1
            else:
                self.roc_list[class_id] = (right_cinfo.top1_rate / left_cinfo.top1_rate)

        self.sum_roc = 0
        self.ave = 0
        num_valid_roc = 0
        
        for roc in self.roc_list:
            if roc >= 0:
                self.sum_roc += roc
                num_valid_roc += 1
        self.ave = self.sum_roc / num_valid_roc

        self.sdev = 0
        for roc in self.roc_list:
            if roc >= 0:
                self.sdev += pow(roc - self.ave, 2.0)
        self.sdev = math.sqrt(self.sdev / num_valid_roc)

        num_roc_under1 = 0
        
        for roc in self.roc_list:
            if roc < 0:
                continue
            if roc < 1:
                num_roc_under1 += 1
        
        self.roc_under1_rate = num_roc_under1 / num_valid_roc
        pass
    
    def write(self, fname):
        with open(fname, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            
            writer.writerow(['roc_under1_rate', self.roc_under1_rate])
            writer.writerow(['ave', self.ave])
            writer.writerow(['sdev', self.sdev])

            writer.writerow(['class_id','label', 'top1_rate', 'top1_rate', 'roc'])
            sorted_indexes = np.argsort(self.roc_list)
            for class_id in sorted_indexes:
                left_cinfo = self.left_einfo.get_class_info(class_id)
                right_cinfo = self.right_einfo.get_class_info(class_id)
                roc = self.roc_list[class_id]
                writer.writerow([class_id, left_cinfo.label, left_cinfo.top1_rate, right_cinfo.top1_rate, roc])

        pass

    def write_left_einfo(self, fname):
        self.left_einfo.write(fname)

    def write_right_einfo(self, fname):
        self.right_einfo.write(fname)
