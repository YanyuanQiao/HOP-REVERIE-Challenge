''' Evaluation of agent trajectories '''

import json
import os
import os.path as osp
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans=None, tok=None):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        with open('data_v2/objpos.json','r') as f:
                self.objpos = json.load(f)

        for split in splits:
            for item in load_datasets([split]):
                if scans is not None and item['scan'] not in scans:
                    continue
                self.gt[str(item['id'])] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%s_%d' % (item['id'], i) for i in range(len(item['instructions']))]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs, self.vppos = load_nav_graphs(self.scans, rtnpos=True)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        self.objProposals, self.obj2viewpoint = loadObjProposals(self.objpos,self.vppos)

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path, ref_objId):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        gt = self.gt[instr_id[:-2]]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        self.scores['trajectory_steps'].append(len(path)-1)

        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )
        # REF sucess or not
        if ref_objId == gt['objId']:
            self.scores['rgs'].append(1)
        else:
            self.scores['rgs'].append(0)
        # navigation - success or not
        end_viewpoint_id = gt['scan'] + '_' + final_position
        if self.objProposals.__contains__(end_viewpoint_id):
            if str(gt['objId']) in self.objProposals[end_viewpoint_id]['objId']:
                self.scores['visible'].append(1)
            else:
                self.scores['visible'].append(0)
        else:
            self.scores['visible'].append(0)
        # navigation - oracle success or not
        oracle_succ = 0
        for passvp in path:
            oracle_viewpoint_id = gt['scan'] + '_' + passvp[0]
            if self.objProposals.__contains__(oracle_viewpoint_id):
                if str(gt['objId']) in self.objProposals[oracle_viewpoint_id]['objId']:
                    oracle_succ = 1
                    break
        self.scores['oracle_visible'].append(oracle_succ)

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file

        print('result length', len(results))
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                self._score_item(item['instr_id'], item['trajectory'], item['predObjId'])

        if 'train' not in self.splits:  # Exclude the training from this. (Because training eval may be partial)
            assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                           % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
            assert len(self.scores['visible']) == len(self.instr_ids)

        score_summary = {
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths'])
        }
        end_successes = sum(self.scores['visible'])
        score_summary['success_rate'] = float(end_successes) / float(len(self.scores['visible']))
        oracle_successes = sum(self.scores['oracle_visible'])
        score_summary['oracle_rate'] = float(oracle_successes) / float(len(self.scores['oracle_visible']))

        spl = [float(visible == 1) * l / max(l, p, 0.01)
            for visible, p, l in
            zip(self.scores['visible'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['spl'] = np.average(spl)

        num_rgs = sum(self.scores['rgs'])
        score_summary['rgs'] = float(num_rgs)/float(len(self.scores['rgs']))

        rgspl = [float(rgsi == 1) * l / max(l, p)
            for rgsi, p, l in
            zip(self.scores['rgs'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['rgspl'] = np.average(rgspl)

        return score_summary, self.scores

    def bleu_score(self, path2inst):
        from bleu import compute_bleu
        refs = []
        candidates = []
        set_trace()
        for path_id, inst in path2inst.items():
            path_id = str(path_id)
            assert path_id in self.gt
            # There are three references
            refs.append([self.tok.split_sentence(sent) for sent in self.gt[path_id]['instructions']])
            candidates.append([self.tok.index_to_word[word_id] for word_id in inst])

        tuple = compute_bleu(refs, candidates, smooth=False)
        bleu_score = tuple[0]
        precisions = tuple[1]

        return bleu_score, precisions

def loadObjProposals(objpos,vppos):
    bboxDir = 'data_v2/BBoxes_v2'
    objProposals = {}
    obj2viewpoint = {}

    for efile in os.listdir(bboxDir):
        if efile.endswith('.json'):
            with open(osp.join(bboxDir, efile)) as f:
                scan = efile.split('_')[0]
                scanvp, _ = efile.split('.')
                data = json.load(f)

                # for a viewpoint (for loop not needed)
                for vp, vv in data.items():
                    # for all visible objects at that viewpoint
                    for objid, objinfo in vv.items():
                        if scanvp not in vppos:
                            continue
                        if objinfo['visible_pos']:
                            distance = ((vppos[scanvp][0]-objpos[scan][objid][0])**2\
                                + (vppos[scanvp][1]-objpos[scan][objid][1])**2\
                                + (vppos[scanvp][2]-objpos[scan][objid][2])**2)**0.5
                            if distance<=3.0:
                                # if such object not already in the dict
                                if obj2viewpoint.__contains__(scan+'_'+objid):
                                    if vp not in obj2viewpoint[scan+'_'+objid]:
                                        obj2viewpoint[scan+'_'+objid].append(vp)
                                else:
                                    obj2viewpoint[scan+'_'+objid] = [vp,]

                                # if such object not already in the dict
                                if objProposals.__contains__(scanvp):
                                    for ii, bbox in enumerate(objinfo['bbox2d']):
                                        objProposals[scanvp]['bbox'].append(bbox)
                                        objProposals[scanvp]['visible_pos'].append(objinfo['visible_pos'][ii])
                                        objProposals[scanvp]['objId'].append(objid)

                                else:
                                    objProposals[scanvp] = {'bbox': objinfo['bbox2d'],
                                                            'visible_pos': objinfo['visible_pos']}
                                    objProposals[scanvp]['objId'] = []
                                    for _ in objinfo['visible_pos']:
                                        objProposals[scanvp]['objId'].append(objid)

    return objProposals, obj2viewpoint

def load_datasets(splits):
    """

    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    import random
    data = []
    old_state = random.getstate()
    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        if "/" not in split:
            with open('data_v2/REVERIE_%s.json' % split) as f:
                new_data = json.load(f)
        else:
            with open(split) as f:
                new_data = json.load(f)

        # Partition
        if number > 0:
            random.seed(0)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data

def load_nav_graphs(scans, rtnpos=False):
    ''' Load connectivity graph for each scan '''
    vppos={}
    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]])
                            vppos[scan+'_'+item['image_id']] = positions[item['image_id']]
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    if rtnpos:
        return graphs, vppos
    else:
        return graphs

def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':
    # eval val_seen and val_unseen
    splits = ['val_seen','val_unseen']
    resfile ='your file path/submit_%s.json'
    for split in splits:
        eval = Evaluation([split])
        score_summary, _ = eval.score(resfile%split)
        print(score_summary)


