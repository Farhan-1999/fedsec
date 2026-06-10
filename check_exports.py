import importlib, traceback
checks = {
    "dtfl.attack": ["Adversary","L0UnsupervisedAttacker","L1FewShotAttacker",
                    "L2PriorAttacker","ModelKnowledge","L3OmniscientAttacker",
                    "DeviceObservation","ObservationFeaturizer","build_observations"],
    "dtfl.metrics": ["capability_advantage","AdvantageResult","AnonymityStats",
                     "anonymity_stats","equivalence_class_sizes","signature_up_to",
                     "LinkabilityCurve","linkability_curve","mean_linkability_at"],
    "dtfl.defense": ["DefenseConfig","BucketMode","CountMode","apply_count_defense",
                     "merge_weight_for","apply_send_delay","apply_padding","PaddingResult",
                     "deadline_quantiles_for","m_min_of","bucketer_for"],
    "dtfl.controller": ["Controller","DeadlinePolicy","FixedEqualWidth","FixedQuantile",
                        "QuantileTrackingController","EWMAQuantileController"],
    "dtfl.learning": ["Dataset","FLModel","NumpySoftmaxModel","get_torch_model",
                      "iid_shard","make_synthetic_classification","load_real_dataset",
                      "FedTrainConfig","FedTrainResult","federated_train"],
    "dtfl.secagg": ["CostConstants","SecAggCost","complete_graph_cost",
                    "sparse_graph_cost","default_degree","SuccessPrediction",
                    "tier_success_probability","hoeffding_success_lower_bound"],
    "dtfl.sim": ["Engine","EngineConfig","SimulationOutput","DeadlinePolicy",
                 "RoundConfig","RoundResult","RoundLatentLog","run_round",
                 "GateReport","evaluate_gates","snr_diagnostic"],
}
import sys; sys.path.insert(0, "src")
bad = False
for mod, names in checks.items():
    try:
        m = importlib.import_module(mod)
    except Exception:
        print(f"CANNOT IMPORT {mod}"); traceback.print_exc(); bad=True; continue
    missing = [n for n in names if not hasattr(m, n)]
    if missing:
        bad = True
        print(f"{mod}: MISSING EXPORTS -> {missing}")
    else:
        print(f"{mod}: OK ({len(names)} exports)")
print("\nALL GOOD" if not bad else "\nFIX THE MISSING EXPORTS ABOVE")
