
import os
import json
import time

import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import utils as us
from sample import Sample


from self_driving.simulation_data import SimulationParams, SimulationData, SimulationDataRecord, SimulationInfo
from self_driving.road_bbox import RoadBoundingBox
from self_driving.vehicle_state_reader import VehicleState
from self_driving.beamng_member import BeamNGMember
from self_driving.decal_road import DecalRoad
import self_driving.beamng_problem as BeamNGProblem
import self_driving.beamng_individual as BeamNGIndividual
import self_driving.beamng_config as cfg
from config import TARGET_THRESHOLD, mlp_range, sdstd_range, curv_range, turncnt_range
from typing import Tuple, List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_data(data: np.ndarray, n_clusters_interval: Tuple[int, int]) -> Tuple[List[int], List[float]]:
    """
    :param data: data to cluster
    :param n_clusters_interval: (min number of clusters, max number of clusters) for silhouette analysis
    :return: list of labels, list of centroid coordinates, optimal silhouette score
    """

    assert n_clusters_interval[0] >= 2, 'Min number of clusters must be >= 2'
    range_n_clusters = np.arange(n_clusters_interval[0], n_clusters_interval[1])
    optimal_score = -1
    optimal_n_clusters = -1
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)  # throws ValueError
        print("For n_clusters = {}, the average silhouette score is: {}".format(n_clusters, silhouette_avg))
        if silhouette_avg > optimal_score:
            optimal_score = silhouette_avg
            optimal_n_clusters = n_clusters

    assert optimal_n_clusters != -1, 'Error in silhouette analysis'
    print('Best score is {} for n_cluster = {}'.format(optimal_score, optimal_n_clusters))

    clusterer = KMeans(n_clusters=optimal_n_clusters).fit(data)
    return clusterer.labels_, clusterer.cluster_centers_



num_cells = 100

def load_data_all(dst):
    inputs = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]: 
                print(".", end='', flush=True)  

                if "ga_" in json_path:
                    y2 = "GA"
                elif "nsga2" in json_path:
                    y2 = "NSGA2"

                y1 = "INPUT"
        
                with open(json_path) as jf:
                    json_data = json.load(jf)

                inputs.append([json_data["sample_nodes"], f"{y2}-{y1}", json_data['misbehaviour'], float(json_data["distance to target"])])
    return inputs


def load_data(dst, i, approach, div):
    inputs = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        if i+approach in subdir and div in subdir and "10h" in subdir:
            for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                    print(".", end='', flush=True)  

                    if "ga_" in json_path:
                        y2 = "GA"
                    elif "nsga2" in json_path:
                        y2 = "NSGA2"

                    y1 = "INPUT"
           
                    with open(json_path) as jf:
                        json_data = json.load(jf)

                    inputs.append([json_data["sample_nodes"], f"{y2}-{y1}", json_data['misbehaviour'], float(json_data["distance to target"])])
    print(len(inputs))
    return inputs


def plot_tSNE(inputs, _folder, features, div, ii=0):
    """
    This function computes diversity using t-sne
    """
    X, y = [], []
    for i in inputs:
        X.append(list(matplotlib.cbook.flatten(i[0])))
        y.append(i[1])

    
    X = np.array(X)

    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    tsne = TSNE(n_components=2, verbose=1, perplexity=0.1, n_iter=3000)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
   
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        # palette=sns.color_palette("hls", n_colors=10),
        data=df,
        legend="full",
        alpha=0.3
    )
    fig.savefig(f"{_folder}/tsne-diag-{features}-{div}-{ii}-0.1.pdf", format='pdf')

    return df


def compute_tSNE_and_cluster_all(inputs, _folder, features, div, ii=0, num=10):

    target_input_ga = 0
    target_input_nsga2 = 0

    input_ga = 0
    input_nsga2 = 0

    for i in inputs:
        if i[3] == 0.0:
            if i[1] == "GA-INPUT":
                target_input_ga += 1
            if i[1] == "NSGA2-INPUT":
                target_input_nsga2 += 1

    if len(inputs) > 3:

        df = plot_tSNE(inputs, _folder, features, div, ii)    
        df = df.reset_index()  # make sure indexes pair with number of rows

        
        np_data_cols = df.iloc[:,691:693]

        n = len(inputs) - 1
        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-0.1.pdf", format='pdf')
        
        df_nsga2 = df[df.label == "NSGA2-INPUT"]
        df_ga = df[df.label =="GA-INPUT"]

        num_clusters = len(centers)

        div_input_ga = df_ga.nunique()['Clusters']/num_clusters
        div_input_nsga2 = df_nsga2.nunique()['Clusters']/num_clusters


        list_data = [("GA", "Input", div_input_ga, len(df_ga.index), target_input_ga),\
             ("NSGA2", "Input", div_input_nsga2, len(df_nsga2.index), target_input_nsga2)]
    
    else:
        for i in inputs:
            if i[1] == "GA-INPUT":
                input_ga += 1
                div_input_ga = 1.0
            if i[1] == "NSGA2-INPUT":
                input_nsga2 += 1
                div_input_nsga2 = 1.0

        list_data = [("GA", "Input", div_input_ga, input_ga, target_input_ga),\
             ("NSGA2", "Input", div_input_nsga2, input_nsga2, target_input_nsga2)]        
                        
    return list_data


def find_best_div_approach(dst, feature_combinations):

    evaluation_area = ["target_cell_in_gray", "target_cell_in_dark"] 
    print(dst)

    for evaluate in evaluation_area:
        
        for features in feature_combinations:
            for i in range(1, 6):
                inputs = []
                for subdir, dirs, files in os.walk(f"{dst}/{evaluate}/{features}", followlinks=False):
                    if features in subdir and str(i)+"-" in subdir and evaluate in subdir and "10h/sim_" in subdir:
                        data_folder = subdir
                        all_inputs = load_data_all(data_folder)
                        inputs = inputs + all_inputs

                list_data = compute_tSNE_and_cluster_all(inputs, f"{dst}/{evaluate}/{features}", features, i)


                
                for data in list_data:
                    dict_data = {
                        "approach": data[0],
                        "diversity": data[1],
                        "run": i,
                        "test input count": data[3],
                        "features": features,
                        "num tsne clusters": str(data[2]),
                        "test input count in target": data[4]
                    }

                    filedest = f"{dst}/{evaluate}/{features}/report_{data[0]}-{data[1]}_{i}.json"
                    with open(filedest, 'w') as f:
                        (json.dump(dict_data, f, sort_keys=True, indent=4))
                

def compute_targets_for_dh(dst, goal, features, threshold, metric):
    count = 0
    samples = []
    _config = cfg.BeamNGConfig()
    _config.name = ""
    problem = BeamNGProblem.BeamNGProblem(_config)
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        for json_path in [os.path.join(subdir, f) for f in files if
                        (
                            f.startswith("simulation") and
                            f.endswith(".json")
                        )]:

            with open(json_path) as jf:
                data = json.load(jf)

            

            nodes = data["road"]["nodes"]
            bbox_size=(-250, 0, 250, 500)
            member = BeamNGMember(data["control_nodes"], nodes, 20, RoadBoundingBox(bbox_size))
            records = data["records"]
            simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
            sim_name = simulation_id # member.config.simulation_name.replace('$(id)', simulation_id)
            simulation_data = SimulationData(sim_name)
            states = []
            for record in records:
                state = VehicleState(timer=record["timer"]
                                    , damage=record["damage"]
                                    , pos=record["pos"]
                                    , dir=record["dir"]
                                    , vel=record["vel"]
                                    , gforces=record["gforces"]
                                    , gforces2=record["gforces2"]
                                    , steering=record["steering"]
                                    , steering_input=record["steering_input"]
                                    , brake=record["brake"]
                                    , brake_input=record["brake_input"]
                                    , throttle=record["throttle"]
                                    , throttle_input=record["throttle_input"]
                                    , throttleFactor=record["throttleFactor"]
                                    , engineThrottle=record["engineThrottle"]
                                    , wheelspeed=record["engineThrottle"]
                                    , vel_kmh=record["engineThrottle"])

                sim_data_record = SimulationDataRecord(**state._asdict(),
                                                    is_oob=record["is_oob"],
                                                    oob_counter=record["oob_counter"],
                                                    max_oob_percentage=record["max_oob_percentage"],
                                                    oob_distance=record["oob_distance"])
                states.append(sim_data_record)

            simulation_data.params = SimulationParams(beamng_steps=data["params"]["beamng_steps"], delay_msec=int(data["params"]["delay_msec"]))
            simulation_data.control_nodes = data["control_nodes"]
            simulation_data.road = DecalRoad.from_dict(data["road"])
            simulation_data.info = SimulationInfo()
            simulation_data.info.start_time = data["info"]["start_time"]
            simulation_data.info.end_time = data["info"]["end_time"]
            simulation_data.info.elapsed_time = data["info"]["elapsed_time"]
            simulation_data.info.success = data["info"]["success"]
            simulation_data.info.computer_name = data["info"]["computer_name"]
            simulation_data.info.id = data["info"]["id"]

            simulation_data.states = states

            if len(states) > 0:
                member.distance_to_boundary = simulation_data.min_oob_distance()
                member.simulation = simulation_data
                misbehaviour = states[-1].is_oob
            else:
                misbehaviour = False
                break
            
            ind = BeamNGIndividual.BeamNGIndividual(member, _config)
            sample = Sample(ind)
            #us.is_oob(sample.ind.sample_nodes, sample.ind.member.simulation.states)
            
            sample.misbehaviour = misbehaviour

            mlp = us.mean_lateral_position(sample)
            stdsa = us.sd_steering(sample)
            curv = us.curvature(sample)
            turncnt = us.segment_count(sample)

            if features == "MeanLateralPosition_Curvature":
                cell = ((mlp/mlp_range), (curv/curv_range))

            if features == "MeanLateralPosition_SegmentCount":
                cell = ((mlp/mlp_range), (turncnt/turncnt_range))

            if features == "MeanLateralPosition_SDSteeringAngle":
                cell = ((mlp/mlp_range), (stdsa/sdstd_range))


            sample.distance_to_target = us.manhattan(cell, goal)

            if sample.distance_to_target <= TARGET_THRESHOLD and misbehaviour == True:
                samples.append(sample)
                count += 1


    samples = sorted(samples, key=lambda x: x.distance_to_target)
    archive = []
    for sample in samples:
        if len(archive) == 0:
            archive.append(sample)
        elif all(us.get_distance_by_metric(a, sample, metric)> threshold for a in archive):
            archive.append(sample)

    target_samples = []
    for sample in archive:
        print(".", end='', flush=True)
        target_samples.append([sample.ind.m.sample_nodes, f"DeepHyperion", sample.misbehaviour, sample.distance_to_target])
    print("DeepHyperion", features, len(target_samples))
    return target_samples


def compute_tSNE_and_cluster_vs_dh(inputs, _folder, features, approach, div, ii=0, num=10):
    if approach == "nsga2":
        approach = "NSGA2"
    elif approach == "ga_":
        approach = "GA"

    target_deepatash = 0
    target_deephyperion = 0

    for i in inputs:
        if i[3] == 0.0:
            if i[1] == f"{approach}-{div}":
                target_deepatash += 1
            if i[1] == "DeepHyperion":
                target_deephyperion += 1

    if len(inputs) > 3:
        df = plot_tSNE(inputs, _folder, features, div, ii)

        df = df.reset_index()  # make sure indexes pair with number of rows
        np_data_cols = df.iloc[:,691:693]


        n = len(inputs) - 1
        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-0.1.pdf", format='pdf')
        
        df_approach = df[df.label == f"{approach}-{div}"]
        df_dh = df[df.label =="DeepHyperion"]

        num_clusters = len(centers)

        div_approach = df_approach.nunique()['Clusters']/num_clusters
        div_deephyperion = df_dh.nunique()['Clusters']/num_clusters

        list_data = [("DeepAtash", div_approach, len(df_approach.index), target_deepatash),
                    ("DeepHyperion", div_deephyperion, len(df_dh.index), target_deephyperion)]

       
    else:
        deepatash = 0
        deephyperion = 0
        dist_deepatash = 0
        dist_deephyperion = 0
        for i in inputs:
            if i[1] == f"{approach}-{div}":
                deepatash += 1
                dist_deepatash = 1
            if i[1] == "DeepHyperion":
                deephyperion += 1
                dist_deephyperion = 1
        
        list_data = [("DeepAtash", dist_deepatash, deepatash, target_deepatash),
                    ("DeepHyperion", dist_deephyperion, deephyperion, target_deephyperion)]

  
    return list_data

def compare_with_dh(approach, div, features, target_area):

    if div == "INPUT":
        threshold = 10.0

    result_folder = f"../experiments/data/beamng/{target_area}"

    
    for feature in features:

        dst = f"../experiments/data/beamng/DeepAtash/{target_area}/{feature[0]}"
        dst_dh = f"../experiments/data/beamng/DeepHyperion/{feature[0]}"

        print(feature)
        for i in range(1, 6):
            for subdir, dirs, files in os.walk(dst_dh, followlinks=False):
                if "dh-"+str(i) in subdir and "beamng_nvidia_runner" in subdir:
                    inputs_dh = compute_targets_for_dh(subdir, feature[1], feature[0], threshold, div)
                    break
            
            inputs_focused = load_data(dst, str(i)+"-", approach, div)



            list_data = compute_tSNE_and_cluster_vs_dh(inputs_dh+inputs_focused, result_folder, feature[0], approach, div)
            

            for data in list_data:
                dict_data = {
                    "approach": data[0],
                    "run": i,
                    "test input count": data[2],
                    "features": feature[0],
                    "num tsne clusters": str(data[1]),
                    "test input count in target": data[3],
                }

                filedest = f"{result_folder}/report_{data[0]}_{feature[0]}_{target_area}_{i}.json"
                with open(filedest, 'w') as f:
                    (json.dump(dict_data, f, sort_keys=True, indent=4))



if __name__ == "__main__": 

    dst = "../experiments/data/beamng/DeepAtash"
    feature_combinations = ["MeanLateralPosition_SegmentCount", "MeanLateralPosition_Curvature", "MeanLateralPosition_SDSteeringAngle"]
    find_best_div_approach(dst, feature_combinations)


    # goal cell for dark area  MLP-TurnCnt (160, 3), MLP-StdSA (162,108), Curv-StdSA (22, 75), MLP-Curv (167, 20)
    # feature and target
    features = [("MeanLateralPosition_SDSteeringAngle",(162/mlp_range, 108/sdstd_range)), ("MeanLateralPosition_Curvature",(167/mlp_range, 20/curv_range)), ("MeanLateralPosition_SegmentCount", (160/mlp_range, 3/turncnt_range))]
    compare_with_dh("nsga2", "INPUT", features, "target_cell_in_dark")

    # goal cell for gray area  MLP-TurnCnt (166, 2), MLP-StdSA (168,90),  MLP-Curv (165, 14)
    # feature and target
    features = [("MeanLateralPosition_SegmentCount", (166/mlp_range, 2/turncnt_range)), ("MeanLateralPosition_SDSteeringAngle",(168/mlp_range, 90/sdstd_range)), ("MeanLateralPosition_Curvature",(165/mlp_range, 14/curv_range))]
    compare_with_dh("nsga2", "INPUT", features, "target_cell_in_gray")