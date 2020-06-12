import random
import numpy as np
from interactions import Interactions
import os 

class Data_process:
    def __init__(self,args):
        data_path = os.getcwd() + '/dataset/'
        ny_checkin_path = data_path + 'toy_IG_ckeckin_dataset.txt'
        ny_demo_path = data_path + 'ny.demo'
        ny_vcat_path = data_path + 'ny.vcat'

        self.ny_vcat = open(ny_vcat_path,'r').readlines()
        self.ny_checkin = open(ny_checkin_path,'r').readlines()
        self.ny_demo = open(ny_demo_path,'r').readlines()

        self.dict_restrict = {
            'year':[2015,2016,2017],
            'num_checkin':41,
            'test_X_num':args.L_hgn,
            'test_Y_num':20,
        }
        self.L_hgn,self.T_hgn = args.L_hgn, args.T_hgn
        self.args = args

    def run(self):
        print('/1')
        index,r_index = 0,0
        dict_locid_catid,dict_catid_to_index,dict_r_catid_to_index= {},{},{}
        for ny_vcat_i in self.ny_vcat[1:]:
            ny_vcat_i = ny_vcat_i.split(',')
            locid_ = ny_vcat_i[0]
            catid_ = ny_vcat_i[1]
            if locid_ not in dict_locid_catid:
                dict_locid_catid[locid_] = (catid_,int(catid_))
                if catid_ not in dict_catid_to_index:
                    dict_catid_to_index[catid_] = index
                    index +=1
                if int(catid_) not in dict_r_catid_to_index:
                    dict_r_catid_to_index[int(catid_)] = r_index
                    r_index +=1
        set_locid_name = set(dict_locid_catid.keys())
        locid_name = list(set(dict_locid_catid.keys()))
        catid_set = list(set(dict_catid_to_index.values()))
        relation_catid_set = list(set(dict_r_catid_to_index.values()))
        print('/2')
        for i in range(len(locid_name)):
            locid_catid_ = dict_locid_catid[locid_name[i]]
            index_ = dict_catid_to_index[locid_catid_[0]]
            r_index_ = dict_r_catid_to_index[locid_catid_[1]]
            dict_locid_catid[locid_name[i]] = (index_,r_index_)
        print('/3')
        dict_uid_category,uid_category_set,gender_set,race_set = {},[],[],[]
        for ny_demo_i in self.ny_demo[1:]:
            ny_demo_i = ny_demo_i.split(',')
            uid_ = ny_demo_i[0]
            gender_ = int(ny_demo_i[1])-1
            race_ = int(ny_demo_i[3])-1
            category_ = gender_*race_ + race_
            dict_uid_category[uid_] = (category_,(gender_,race_))
            uid_category_set.append(category_)
            gender_set.append(gender_)
            race_set.append(race_)
        uid_category_set = list(set(uid_category_set))
        uid_name = set(dict_uid_category.keys())
        self.gender_set = list(set(gender_set))
        self.race_set = list(set(race_set))
        print('/4')
        dict_uid_time_locid = {}
        for i in range(len(self.ny_checkin)-1):
            ny_checkin_i = self.ny_checkin[i+1].split(',')
            uid_,time_,locid_ = ny_checkin_i[1],ny_checkin_i[2],ny_checkin_i[5].strip('\n')
            time_day,time_hms = time_.split()[0],time_.split()[1]
            if uid_ in uid_name and locid_ in set_locid_name:
                year_ = int(time_day.split('-')[0])
                if year_ in self.dict_restrict['year']:
                    if uid_ not in dict_uid_time_locid:
                        dict_uid_time_locid[uid_] = [(time_day,time_hms,locid_)]
                    else:
                        dict_uid_time_locid[uid_].append((time_day,time_hms,locid_))
        print('/5')
        uid_name = list(set(dict_uid_time_locid.keys()))
        dict_uid_locid_relation,dict_locid_to_entity,index = {},{},0
        for u_i in range(len(uid_name)):
            uid_time_locid_ = dict_uid_time_locid[uid_name[u_i]]
            if len(uid_time_locid_) >= self.dict_restrict['num_checkin']:
                sorted_uid_time_locid_ = sorted(uid_time_locid_)
                for i in range(len(sorted_uid_time_locid_)):
                    time_day_,time_hms_,locid_ = sorted_uid_time_locid_[i][0],sorted_uid_time_locid_[i][1],sorted_uid_time_locid_[i][2]
                    daynum_time_day_ = int(time_day_.split('-')[1])-1
                    time_h = int(time_hms_.split('-')[0].split(':')[0])
                    if time_h >11:
                        time_h = 1
                    else:
                        time_h = 0
                    relation_time = time_h * daynum_time_day_ + daynum_time_day_
                    sorted_uid_time_locid_[i] = (locid_,relation_time)
                    if locid_ not in dict_locid_to_entity:
                        dict_locid_to_entity[locid_] = index   
                        index +=1
                dict_uid_locid_relation[uid_name[u_i]] = sorted_uid_time_locid_
        uid_name = list(set(dict_uid_locid_relation.keys()))
        locid_name = list(set(dict_locid_to_entity.keys()))
        print('/6')
        test_X_num = self.dict_restrict['test_X_num']
        test_Y_num = self.dict_restrict['test_Y_num']
        train_set,test_set,test_X_set,test_Y_set,relation_time_set,data_set = [],[],[],[],[],[]
        for i in range(len(uid_name)):
            uid_locid_relation_ = dict_uid_locid_relation[uid_name[i]]
            data_set_,catid_set_ = [],[]
            for j in range(len(uid_locid_relation_)):
                locid_ = uid_locid_relation_[j][0]
                catid_ = dict_locid_catid[locid_][0]
                entity_locid_ = dict_locid_to_entity[locid_]
                relation_time_ = uid_locid_relation_[j][1]
                data_set_.append(entity_locid_)
                relation_time_set.append(relation_time_)
                catid_set_.append(catid_)
            data_set.append(data_set_)
            train_set_ = data_set_[:len(data_set_)-test_X_num-test_Y_num]
            test_X_set_ = data_set_[len(data_set_)-test_X_num-test_Y_num:len(data_set_)-test_Y_num]
            test_Y_set_ = data_set_[len(data_set_)-test_Y_num:]
            test_set_ = data_set_[len(data_set_)-test_X_num-test_Y_num:]
            train_set.append(train_set_)
            test_set.append(test_set_)
            test_X_set.append(test_X_set_)
            test_Y_set.append(test_Y_set_)
        self.test_X_set = np.array(test_X_set)
        relation_time_set = list(set(relation_time_set))
        entity_set = list(set(dict_locid_to_entity.values()))
        print('/7')
        dict_uid_category_to_entity = {}
        for i in range(len(uid_category_set)):
            dict_uid_category_to_entity[uid_category_set[i]] = uid_category_set[i]+len(entity_set)
        dict_catid_to_entity = {}
        for i in range(len(catid_set)):
            dict_catid_to_entity[catid_set[i]] = len(entity_set)+len(uid_category_set)+catid_set[i]
        dict_relation_catid_to_r = {}
        for i in range(len(relation_catid_set)):
            dict_relation_catid_to_r[relation_catid_set[i]] = len(relation_time_set)+relation_catid_set[i]
        print('/8')
        self.dict_KG,self.uid_attribute_category = {},[]
        for i in range(len(uid_name)):
            uid_category_,uid_category_detail_ = dict_uid_category[uid_name[i]]
            entity_uid_category_ = dict_uid_category_to_entity[uid_category_]
            uid_locid_relation_ = dict_uid_locid_relation[uid_name[i]]
            gender_,race_ = uid_category_detail_[0],uid_category_detail_[1]
            self.uid_attribute_category.append((gender_,race_))
            for j in range(len(uid_locid_relation_)):
                locid_ = uid_locid_relation_[j][0]
                relation_time_ = uid_locid_relation_[j][1] 
                entity_locid_ = dict_locid_to_entity[locid_]
                catid_,relation_catid_ = dict_locid_catid[locid_][0],dict_locid_catid[locid_][1]
                entity_catid_ = dict_catid_to_entity[catid_]
                relation_catid_ = dict_relation_catid_to_r[relation_catid_]
                if entity_locid_ not in self.dict_KG:
                    self.dict_KG[entity_locid_] = {}
                    for k in range(len(self.gender_set)):
                        self.dict_KG[entity_locid_]['gender'+'-'+str(self.gender_set[k])] = []
                        if self.gender_set[k] == gender_:
                            self.dict_KG[entity_locid_]['gender'+'-'+str(self.gender_set[k])].append([entity_locid_,relation_time_,entity_uid_category_])
                            self.dict_KG[entity_locid_]['gender'+'-'+str(self.gender_set[k])].append([entity_locid_,relation_catid_,entity_catid_])
                    for k in range(len(self.race_set)):
                        self.dict_KG[entity_locid_]['race'+'-'+str(self.race_set[k])] = []
                        if self.race_set[k] == race_:
                            self.dict_KG[entity_locid_]['race'+'-'+str(self.race_set[k])].append([entity_locid_,relation_time_,entity_uid_category_])
                            self.dict_KG[entity_locid_]['race'+'-'+str(self.race_set[k])].append([entity_locid_,relation_catid_,entity_catid_])
                else:
                    for k in range(len(self.gender_set)):
                        if self.gender_set[k] == gender_:
                            self.dict_KG[entity_locid_]['gender'+'-'+str(self.gender_set[k])].append([entity_locid_,relation_time_,entity_uid_category_])
                            self.dict_KG[entity_locid_]['gender'+'-'+str(self.gender_set[k])].append([entity_locid_,relation_catid_,entity_catid_])
                    for k in range(len(self.race_set)):
                        if self.race_set[k] == race_:
                            self.dict_KG[entity_locid_]['race'+'-'+str(self.race_set[k])].append([entity_locid_,relation_time_,entity_uid_category_])
                            self.dict_KG[entity_locid_]['race'+'-'+str(self.race_set[k])].append([entity_locid_,relation_catid_,entity_catid_])

        self.dict_itemid_upfdf = dict()
        for i in range(len(uid_name)):
            uid_upf_ = self.uid_attribute_category[i]
            uid_data_set_ = data_set[i]
            for j in range(len(uid_data_set_)):
                if uid_data_set_[j] not in self.dict_itemid_upfdf:
                    self.dict_itemid_upfdf[uid_data_set_[j]] = list()
                self.dict_itemid_upfdf[uid_data_set_[j]].append(uid_upf_)
        itemid_name = list(set(self.dict_itemid_upfdf.keys()))
        for i in range(len(itemid_name)):
            itemid_upfdf_ = self.dict_itemid_upfdf[itemid_name[i]]
            b_0,b_1,b_2,g_0,g_1,g_2 = 0,0,0,0,0,0
            for j in range(len(itemid_upfdf_)):
                if itemid_upfdf_[j] == (0,0):
                    b_0 +=1
                elif itemid_upfdf_[j] == (0,1):
                    b_1 +=1
                elif itemid_upfdf_[j] == (0,2):
                    b_2 +=1
                elif itemid_upfdf_[j] == (1,0):
                    g_0 +=1
                elif itemid_upfdf_[j] == (1,1):
                    g_1 +=1
                elif itemid_upfdf_[j] == (1,2):
                    g_2 +=1
                else:
                    print('Error!!')
            b_0_p,b_1_p,b_2_p = b_0/len(itemid_upfdf_),b_1/len(itemid_upfdf_),b_2/len(itemid_upfdf_)
            g_0_p,g_1_p,g_2_p = g_0/len(itemid_upfdf_),g_1/len(itemid_upfdf_),g_2/len(itemid_upfdf_)
            self.dict_itemid_upfdf[itemid_name[i]] = [b_0_p,b_1_p,b_2_p,g_0_p,g_1_p,g_2_p]

        num_user = len(uid_name)
        num_item = len(entity_set)
        n_entities = len(entity_set)+len(uid_category_set)+len(catid_set)
        n_relation = len(relation_time_set)+len(relation_catid_set)

        train = Interactions(train_set, num_user, num_item)
        train.to_sequence(self.L_hgn, self.T_hgn)

        sequences_np = train.sequences.sequences
        targets_np = train.sequences.targets
        users_np = train.sequences.user_ids
        train_matrix = train.tocsr() 

        param_ = [self.args,num_user,num_item,n_entities,n_relation]
        data_ = [users_np,sequences_np,targets_np,train_matrix]
        test_ = [train,self.test_X_set,test_Y_set,self.uid_attribute_category]       

        train_data,test_data = list(),list()
        entity_set = set(entity_set)
        for i in range(len(train_set)):
            uid_ = i
            neg_entity_set_ = list(entity_set - set(train_set[i]))
            neg_train_set_ = random.sample(neg_entity_set_,len(train_set[i]))
            for j in range(len(train_set[i])):
                train_data.append([uid_,train_set[i][j],1])
                train_data.append([uid_,neg_train_set_[j],0])
        for i in range(len(test_set)):
            uid_ = i
            neg_entity_set_ = list(entity_set - set(test_set[i]))
            neg_test_set_ = random.sample(neg_entity_set_,len(test_set[i]))
            for j in range(len(test_set[i])):
                test_data.append([uid_,test_set[i][j],1])
                test_data.append([uid_,neg_test_set_[j],0])    
        train_data = np.array(train_data) 
        eval_data = train_data 
        test_data = np.array(test_data)  

        user_history_dict = dict() 
        for i in range(len(data_set)):
            uid_ = i
            user_history_dict[uid_] = data_set[i]
        
        pickle_data = {
            'train_data':train_data,
            'eval_data':eval_data,
            'test_data':test_data,
            'n_entity':n_entities,
            'n_relation':n_relation,
            'kg_np':self.KG_random_generator(),
            'user_history_dict':user_history_dict,
        }
        return param_,data_,test_
    def KG_random_generator_(self):
        dict_KG = self.dict_KG
        list_entity = list(set(dict_KG.keys()))
        KG = []
        for i in range(len(list_entity)):
            attribute_category = list(dict_KG[list_entity[i]].keys())
            for j in range(len(attribute_category)):  
                KG_ = dict_KG[list_entity[i]][attribute_category[j]]  
                KG += KG_  
        KG = np.array(KG)
        return KG
    
    def KG_random_generator(self):
        dict_KG = self.dict_KG
        list_entity = list(set(dict_KG.keys()))
        KG = dict()
        for i in range(len(list_entity)):
            if list_entity[i] not in KG:
                KG[list_entity[i]] = list()    
            attribute_category = list(dict_KG[list_entity[i]].keys())   
            key = True
            while key:
                random_attribute_category = random.choice(attribute_category)
                KG_ = dict_KG[list_entity[i]][random_attribute_category]
                if len(KG_) !=0:
                    for j in range(len(KG_)):
                        KG[list_entity[i]].append(KG_[j])
                    key = False
            KG[list_entity[i]] = np.array(KG[list_entity[i]])
        return KG


















