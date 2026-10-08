"""Command line entry point for the detector-agnostic provenance-graph attack.

Examples
--------
python ATTACK.py --dataset cadets --detector stgan
python ATTACK.py --dataset trace --detector flash --top-k 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


# Keep direct interactive runs on the same UTF-8 encoding as persisted logs.
# This prevents Chinese diagnostics from passing through a legacy Windows
# console encoding when ATTACK.py is not launched by the experiment driver.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
# attack_directed_topk was originally written as a script and uses local imports.
# Preserve that module's import contract without changing its attack logic.
MAGIC_ROOT = ROOT / "MAGIC_main"
if str(MAGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MAGIC_ROOT))

from Detector_Factory import create_detector, get_supported_detectors
from MAGIC_main.attack_directed_topk import DirectedTopKAttackAgent
from MAGIC_main.experiment_utils import (
    attach_run_log,
    configure_reproducibility,
    create_run_directory,
    save_experiment_artifacts,
)
from MAGIC_main.utils.loaddata import load_entity_level_dataset, load_metadata


logger = logging.getLogger("attack.experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run directed Top-K provenance-graph attack against a detector adapter."
    )
    parser.add_argument("--dataset", required=True, choices=("cadets", "theia", "trace"))
    parser.add_argument("--detector", required=True, choices=get_supported_detectors())
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--test-index", type=int, default=0)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument(
        "--action-policy",
        choices=(
            "ppo_online", "ppo_frozen", "random", "greedy",
            "linucb", "double_dqn", "benign_mimicry", "no_attack",
        ),
        default="ppo_online",
        help=("Action strategy: online PPO, frozen PPO, random, query-fair "
              "adaptive greedy, LinUCB, Double DQN, detector-observable "
              "Benign-Mimicry, or clean no-attack baseline."),
    )
    parser.add_argument(
        "--ppo-selection-mode",
        choices=("auto", "sample", "argmax"),
        default="auto",
        help=("PPO action selection: auto uses sample for online PPO and "
              "argmax for frozen PPO; frozen sample remains available as "
              "an explicit stochastic evaluation."),
    )
    parser.add_argument(
        "--target-policy",
        choices=(
            "threshold_margin", "highest_score", "random_alert",
            "hybrid_margin_random", "initial_tp_cycle", "initial_tp_margin",
            "initial_tp_random",
        ),
        default="threshold_margin",
        help=("Current-alert target ranking strategy. initial_tp_cycle, "
              "initial_tp_margin, and initial_tp_random are label-assisted "
              "evaluation protocols "
              "restricted to the detector's initial direct true-positive "
              "nodes. initial_tp_margin ranks that fixed pool by threshold "
              "margin; initial_tp_random samples it without replacement "
              "using --seed."),
    )
    parser.add_argument(
        "--target-margin-fraction",
        type=float,
        default=0.5,
        help=("Fraction of each target batch selected by threshold margin "
              "when --target-policy hybrid_margin_random is used."),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate-policy",
        choices=("semantic_mixed", "random_legal"),
        default="semantic_mixed",
        help=("Candidate selection: proposed semantic local/global mixture, or "
              "random sampling from the semantically legal pool for ablation."),
    )
    parser.add_argument(
        "--reward-mode",
        choices=("full", "none", "score_only", "no_graph_penalty"),
        default="full",
        help="Reward preset used by PPO/greedy feedback and reward ablations.",
    )
    parser.add_argument("--reward-score-weight", type=float, default=None)
    parser.add_argument("--reward-crossing-weight", type=float, default=None)
    parser.add_argument("--reward-indirect-weight", type=float, default=None)
    parser.add_argument("--reward-perturbation-weight", type=float, default=None)
    parser.add_argument(
        "--experiment-variant",
        choices=(
            "auto", "full_model", "no_reward", "score_only",
            "no_graph_penalty", "no_candidate_selector", "no_ppo",
            "linucb_baseline", "double_dqn_baseline",
            "benign_mimicry_baseline",
            "linucb_no_context", "linucb_no_exploration",
            "linucb_no_candidate_selector", "linucb_no_macro_action",
            "linucb_no_reward_feedback", "orthrus_full_aware",
        ),
        default="auto",
        help="Explicit ablation label stored in summaries and paper tables.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Request deterministic PyTorch/CUDA kernels (unsupported kernels warn).",
    )
    parser.add_argument(
        "--threshold-source",
        choices=("native", "fixed"),
        default="native",
        help=("Detector threshold source: native evaluation reproduction (default), "
              "or an explicit fixed threshold."),
    )
    parser.add_argument("--threshold", type=float, help="Required with --threshold-source fixed.")
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        help=("Optional target-detector checkpoint override. This is separate "
              "from --checkpoint, which is reserved for the PPO policy."),
    )
    parser.add_argument(
        "--load-checkpoint", action="store_true",
        help="Load the PPO checkpoint before attacking.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="PPO checkpoint path. A detector/dataset-specific path is used by default.",
    )
    parser.add_argument(
        "--warmup-updates", type=int,
        help="Uniform-exploration PPO updates before policy sampling; default is 0.",
    )
    parser.add_argument(
        "--linucb-alpha", type=float, default=1.0,
        help="LinUCB exploration coefficient (default: 1.0).",
    )
    parser.add_argument(
        "--linucb-context-mode",
        choices=("full", "constant"),
        default="full",
        help=("LinUCB context: full uses the attack state; constant removes "
              "state information for the context ablation."),
    )
    parser.add_argument(
        "--macro-action-length",
        type=int,
        default=None,
        help=("Same-semantic edge insertions per detector query. The framework "
              "default is 5; use 1 for the no-macro-action ablation."),
    )
    parser.add_argument(
        "--ddqn-batch-size", type=int, default=16,
        help="Double DQN replay minibatch size (default: 16).",
    )
    parser.add_argument(
        "--clean-repeat-checks", type=int, default=0,
        help="Extra clean-graph detector queries used to audit score reproducibility.",
    )
    parser.add_argument(
        "--threshold-epsilons", type=float, nargs="*",
        default=(1e-6, 1e-3, 1e-1),
        help="Absolute threshold perturbations used for final sensitivity reporting.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "attack_results",
        help="Parent directory for structured experiment artifacts.",
    )
    parser.add_argument("--run-name", help="Optional explicit experiment directory name.")
    parser.add_argument(
        "--no-save-artifacts", action="store_true",
        help="Disable summary, score, trajectory, and modification persistence.",
    )
    parser.add_argument(
        "--no-generate-plots", action="store_true",
        help="Do not generate per-run paper figures after saving artifacts.",
    )
    parser.add_argument(
        "--save-final-graph", action="store_true",
        help="Also save the large final DGL graph; modifications are always saved otherwise.",
    )
    parser.add_argument("--max-queries", type=int, help="Observable detector-query budget.")
    parser.add_argument("--max-added-edges", type=int, help="Global added-edge budget.")
    parser.add_argument("--max-runtime-minutes", type=float, help="Wall-clock attack budget.")
    parser.add_argument(
        "--no-improvement-patience", type=int,
        help="Stop after this many consecutive targets without an observable score decrease.",
    )
    parser.add_argument(
        "--flash-semantic-state",
        "--flash-semantic-json",
        dest="flash_semantic_state",
        help=(
            "Required for --detector flash: native .npz semantic/topology "
            "state (legacy JSON is accepted for compatibility)."
        ),
    )
    parser.add_argument(
        "--stgan-semantic-state",
        help=(
            "Required for --detector stgan: native .npz CDM topology, time, "
            "entity-label and STGAN Word2Vec state."
        ),
    )
    parser.add_argument(
        "--orthrus-attack-state",
        help=(
            "Orthrus E3 attack-state .npz. When omitted, the adapter selects "
            "the dataset-specific CADETS or THEIA state under "
            "Orthrus_official/artifacts/attack_state/."
        ),
    )
    parser.add_argument(
        "--orthrus-full-aware",
        action="store_true",
        help=(
            "Enable the Orthrus full-mode temporal attack: anchor edits before "
            "the native maximum-loss edge, rank candidates with native edge "
            "loss, roll back harmful macros, and protect escaped initial TP nodes."
        ),
    )
    parser.add_argument(
        "--orthrus-candidate-probe-size",
        type=int,
        default=20,
        help="Number of legal endpoints screened with native Orthrus edge loss per macro.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clean_repeat_checks < 0:
        raise ValueError("--clean-repeat-checks must be >= 0")
    if any(epsilon < 0 for epsilon in args.threshold_epsilons):
        raise ValueError("--threshold-epsilons values must be >= 0")
    if args.warmup_updates is not None and args.warmup_updates < 0:
        raise ValueError("--warmup-updates must be >= 0")
    if args.linucb_alpha < 0:
        raise ValueError("--linucb-alpha must be >= 0")
    if args.orthrus_candidate_probe_size <= 0:
        raise ValueError("--orthrus-candidate-probe-size must be > 0")
    if args.orthrus_full_aware:
        if args.detector != "orthrus":
            raise ValueError("--orthrus-full-aware requires --detector orthrus")
        if args.action_policy != "linucb":
            raise ValueError(
                "--orthrus-full-aware requires --action-policy linucb so "
                "subject and file policies can be learned separately"
            )
    if args.macro_action_length is not None and args.macro_action_length <= 0:
        raise ValueError("--macro-action-length must be > 0")
    if (
        args.linucb_context_mode != "full"
        and args.action_policy != "linucb"
    ):
        raise ValueError(
            "--linucb-context-mode is only meaningful for --action-policy linucb"
        )
    if args.ddqn_batch_size <= 0:
        raise ValueError("--ddqn-batch-size must be > 0")
    if args.action_policy == "ppo_online" and args.ppo_selection_mode == "argmax":
        raise ValueError(
            "ppo_online requires sample selection for on-policy exploration"
        )
    if (
        args.action_policy not in {"ppo_online", "ppo_frozen"}
        and args.ppo_selection_mode != "auto"
    ):
        raise ValueError(
            "--ppo-selection-mode is only meaningful for PPO policies"
        )
    if args.ppo_selection_mode == "auto":
        if args.action_policy == "ppo_online":
            ppo_selection_mode = "sample"
        elif args.action_policy == "ppo_frozen":
            ppo_selection_mode = "argmax"
        else:
            ppo_selection_mode = "not_applicable"
    else:
        ppo_selection_mode = args.ppo_selection_mode
    if (args.warmup_updates not in (None, 0)
            and args.action_policy != "ppo_online"):
        raise ValueError("--warmup-updates is only meaningful for ppo_online")
    if args.load_checkpoint and args.checkpoint is None:
        raise ValueError(
            "--load-checkpoint requires an explicit --checkpoint path for reproducibility"
        )
    if args.action_policy == "ppo_frozen" and not args.load_checkpoint:
        raise ValueError("--action-policy ppo_frozen requires --load-checkpoint")
    if args.action_policy in {
        "random", "greedy", "linucb", "double_dqn",
        "benign_mimicry", "no_attack"
    } and args.load_checkpoint:
        raise ValueError(
            f"--action-policy {args.action_policy} cannot be combined with --load-checkpoint"
        )
    reward_presets = {
        "full": (1.0, 0.5, 0.2, 0.05),
        "none": (0.0, 0.0, 0.0, 0.0),
        "score_only": (1.0, 0.0, 0.0, 0.0),
        "no_graph_penalty": (1.0, 0.5, 0.2, 0.0),
    }
    reward_values = list(reward_presets[args.reward_mode])
    explicit_reward_values = (
        args.reward_score_weight,
        args.reward_crossing_weight,
        args.reward_indirect_weight,
        args.reward_perturbation_weight,
    )
    for index, value in enumerate(explicit_reward_values):
        if value is not None:
            if value < 0:
                raise ValueError("reward weights must be >= 0")
            reward_values[index] = float(value)
    (
        score_weight,
        crossing_weight,
        indirect_weight,
        perturbation_weight,
    ) = reward_values

    if args.experiment_variant == "auto":
        if args.orthrus_full_aware:
            experiment_variant = "orthrus_full_aware"
        elif args.action_policy == "no_attack":
            experiment_variant = "no_attack"
        elif args.action_policy == "random":
            experiment_variant = "no_ppo"
        elif args.action_policy == "greedy":
            experiment_variant = "greedy"
        elif (
            args.action_policy == "linucb"
            and args.linucb_context_mode == "constant"
        ):
            experiment_variant = "linucb_no_context"
        elif args.action_policy == "linucb" and args.linucb_alpha == 0:
            experiment_variant = "linucb_no_exploration"
        elif (
            args.action_policy == "linucb"
            and args.candidate_policy == "random_legal"
        ):
            experiment_variant = "linucb_no_candidate_selector"
        elif (
            args.action_policy == "linucb"
            and args.macro_action_length == 1
        ):
            experiment_variant = "linucb_no_macro_action"
        elif args.action_policy == "linucb" and args.reward_mode == "none":
            experiment_variant = "linucb_no_reward_feedback"
        elif args.action_policy == "linucb":
            experiment_variant = "linucb_baseline"
        elif args.action_policy == "double_dqn":
            experiment_variant = "double_dqn_baseline"
        elif args.action_policy == "benign_mimicry":
            experiment_variant = "benign_mimicry_baseline"
        elif args.candidate_policy == "random_legal":
            experiment_variant = "no_candidate_selector"
        elif args.reward_mode == "none":
            experiment_variant = "no_reward"
        elif args.reward_mode == "score_only":
            experiment_variant = "score_only"
        elif args.reward_mode == "no_graph_penalty":
            experiment_variant = "no_graph_penalty"
        else:
            experiment_variant = "full_model"
    else:
        experiment_variant = args.experiment_variant
    variant_requirements = {
        "no_reward": (args.reward_mode == "none", "--reward-mode none"),
        "score_only": (args.reward_mode == "score_only", "--reward-mode score_only"),
        "no_graph_penalty": (
            args.reward_mode == "no_graph_penalty",
            "--reward-mode no_graph_penalty",
        ),
        "no_candidate_selector": (
            args.candidate_policy == "random_legal",
            "--candidate-policy random_legal",
        ),
        "no_ppo": (args.action_policy == "random", "--action-policy random"),
        "linucb_no_context": (
            args.action_policy == "linucb"
            and args.linucb_context_mode == "constant",
            "--action-policy linucb --linucb-context-mode constant",
        ),
        "linucb_no_exploration": (
            args.action_policy == "linucb" and args.linucb_alpha == 0,
            "--action-policy linucb --linucb-alpha 0",
        ),
        "linucb_no_candidate_selector": (
            args.action_policy == "linucb"
            and args.candidate_policy == "random_legal",
            "--action-policy linucb --candidate-policy random_legal",
        ),
        "linucb_no_macro_action": (
            args.action_policy == "linucb" and args.macro_action_length == 1,
            "--action-policy linucb --macro-action-length 1",
        ),
        "linucb_no_reward_feedback": (
            args.action_policy == "linucb" and args.reward_mode == "none",
            "--action-policy linucb --reward-mode none",
        ),
    }
    if experiment_variant in variant_requirements:
        valid, required = variant_requirements[experiment_variant]
        if not valid:
            raise ValueError(
                f"experiment variant {experiment_variant} requires {required}"
            )
    reproducibility = configure_reproducibility(args.seed, args.deterministic)
    run_dir = None
    if not args.no_save_artifacts:
        run_dir = create_run_directory(
            args.output_dir,
            detector=args.detector,
            dataset=args.dataset,
            action_policy=args.action_policy,
            target_policy=args.target_policy,
            seed=args.seed,
            run_name=args.run_name,
        )
        log_path = attach_run_log(run_dir)
        logger.info(f"[Experiment] run_dir={run_dir}, log={log_path}")
    logger.info(
        f"[Experiment] seed={args.seed}, deterministic={args.deterministic}, "
        f"action_policy={args.action_policy}, target_policy={args.target_policy}"
    )

    # Orthrus uses its own compact state assembled exclusively from the native
    # Orthrus artifacts and evaluator output.  Do not consult the MAGIC/FLASH
    # metadata path for this detector.
    if args.detector == "orthrus":
        if args.test_index != 0:
            raise ValueError("Orthrus E3 attack states require --test-index 0")
    else:
        metadata = load_metadata(args.dataset)
        if not 0 <= args.test_index < metadata["n_test"]:
            raise ValueError(
                f"--test-index must be in [0, {metadata['n_test'] - 1}]"
            )

    detector_kwargs = {"dataset_name": args.dataset, "top_k": args.top_k}
    if args.device is not None:
        detector_kwargs["device"] = args.device
    if args.detector in {"magic", "flash", "orthrus", "provguard", "stgan"}:
        detector_kwargs["threshold_source"] = args.threshold_source
        detector_kwargs["threshold"] = args.threshold
        if args.threshold_source == "fixed" and args.threshold is None:
            raise ValueError("--threshold-source fixed requires --threshold")
    if args.detector_checkpoint is not None:
        detector_kwargs["checkpoint_path"] = str(args.detector_checkpoint)
    if args.detector == "orthrus":
        detector_kwargs["attack_state_path"] = args.orthrus_attack_state
    detector = create_detector(args.detector, **detector_kwargs)
    detector.set_evaluation_context(args.test_index)
    if args.detector == "flash":
        if not args.flash_semantic_state:
            raise ValueError("--flash-semantic-state is required when --detector flash")
        from Detector.Flash_State import FlashSemanticState
        detector.attach_semantic_state(
            FlashSemanticState.load(args.flash_semantic_state)
        )
    elif args.detector == "stgan":
        if not args.stgan_semantic_state:
            raise ValueError("--stgan-semantic-state is required when --detector stgan")
        from Detector.STGAN_State import STGANSemanticState
        detector.attach_semantic_state(
            STGANSemanticState.load(args.stgan_semantic_state)
        )
    elif args.detector == "orthrus":
        from Detector.Orthrus_State import OrthrusAttackState
        detector.attach_attack_state(
            OrthrusAttackState.load(detector.attack_state_path)
        )

    agent = DirectedTopKAttackAgent(
        dataset_name=args.dataset,
        top_k=args.top_k,
        detector=detector,
        action_policy=args.action_policy,
        target_policy=args.target_policy,
        target_margin_fraction=args.target_margin_fraction,
        seed=args.seed,
        candidate_policy=args.candidate_policy,
        experiment_variant=experiment_variant,
        ppo_selection_mode=ppo_selection_mode,
        orthrus_full_aware=args.orthrus_full_aware,
    )
    agent.orthrus_candidate_probe_size = args.orthrus_candidate_probe_size
    agent.reward_mode = args.reward_mode
    agent.config.SCORE_DROP_WEIGHT = score_weight
    agent.config.THRESHOLD_CROSS_WEIGHT = crossing_weight
    agent.config.INDIRECT_WEIGHT = indirect_weight
    agent.config.PERTURBATION_WEIGHT = perturbation_weight
    agent.reward_config = {
        "mode": args.reward_mode,
        "score_weight": score_weight,
        "crossing_weight": crossing_weight,
        "indirect_weight": indirect_weight,
        "perturbation_weight": perturbation_weight,
    }
    logger.info(
        f"[Experiment] variant={experiment_variant}, "
        f"candidate_policy={args.candidate_policy}, "
        f"target_margin_fraction={args.target_margin_fraction}, "
        f"linucb_context_mode={args.linucb_context_mode}, "
        f"linucb_alpha={args.linucb_alpha}, "
        f"macro_action_length={args.macro_action_length or 'default'}, "
        f"ppo_selection_mode={ppo_selection_mode}, "
        f"reward={agent.reward_config}"
    )
    agent.clean_repeat_checks = args.clean_repeat_checks
    agent.threshold_sensitivity_epsilons = tuple(args.threshold_epsilons)
    if args.max_rounds is not None:
        agent.config.MAX_ROUNDS = args.max_rounds
    if args.macro_action_length is not None:
        if agent.config.MAX_STEPS_PER_NODE % args.macro_action_length != 0:
            raise ValueError(
                "MAX_STEPS_PER_NODE must be divisible by --macro-action-length"
            )
        agent.config.MACRO_ACTION_LENGTH = args.macro_action_length
    if args.warmup_updates is not None:
        agent.config.WARMUP_UPDATES = args.warmup_updates
        agent.ppo_agent.config.WARMUP_UPDATES = args.warmup_updates
    agent.config.LINUCB_ALPHA = args.linucb_alpha
    if agent.linucb_policy is not None:
        agent.linucb_policy.alpha = args.linucb_alpha
        agent.linucb_policy.context_mode = args.linucb_context_mode
    agent.config.DDQN_BATCH_SIZE = args.ddqn_batch_size
    if agent.double_dqn_policy is not None:
        agent.double_dqn_policy.batch_size = args.ddqn_batch_size

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if run_dir is not None and args.action_policy == "ppo_online":
            checkpoint_path = run_dir / "ppo_checkpoint.pth"
        elif run_dir is not None and args.action_policy in {
            "linucb", "double_dqn"
        }:
            checkpoint_path = (
                run_dir / f"{args.action_policy}_checkpoint.pth"
            )
        else:
            checkpoint_path = ROOT / (
                f"ppo_checkpoint_{args.detector}_{args.dataset}_"
                f"{args.action_policy}_seed{args.seed}.pth"
            )
    agent.checkpoint_path = str(checkpoint_path)
    if args.load_checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_path}")
        agent.ppo_agent.load_checkpoint(str(checkpoint_path))
        print(
            f"Loaded PPO checkpoint: {checkpoint_path} "
            f"(updates={agent.ppo_agent.training_stats.get('total_updates', 0)})"
        )
    overrides = {
        "MAX_QUERIES": args.max_queries,
        "MAX_TOTAL_ADDED_EDGES": args.max_added_edges,
        "MAX_RUNTIME_SECONDS": (None if args.max_runtime_minutes is None
                                else args.max_runtime_minutes * 60.0),
        "NO_IMPROVEMENT_PATIENCE": args.no_improvement_patience,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(agent.config, name, value)

    if args.detector == "orthrus":
        test_graph = detector.state.to_dgl()
    else:
        test_graph = load_entity_level_dataset(args.dataset, "test", args.test_index)
    try:
        result = agent.attack(test_graph)
    finally:
        close_detector = getattr(detector, "close", None)
        if callable(close_detector):
            close_detector()
    result["checkpoint_path"] = str(checkpoint_path)
    result["run_dir"] = None if run_dir is None else str(run_dir)
    if run_dir is not None:
        artifact_paths = save_experiment_artifacts(
            run_dir,
            args=args,
            reproducibility=reproducibility,
            result=result,
            agent=agent,
            save_final_graph=args.save_final_graph,
        )
        logger.info(f"[Experiment] artifacts={artifact_paths}")
        if not args.no_generate_plots:
            try:
                from MAGIC_main.paper_outputs import generate_single_run_outputs

                figures = generate_single_run_outputs(run_dir, project_root=ROOT)
                logger.info(
                    f"[Experiment] per-run figures={figures}, "
                    f"directory={run_dir / 'figures'}"
                )
            except ImportError as error:
                logger.warning(
                    "[Experiment] 跳过自动出图：缺少绘图库依赖。"
                    f"可使用 --no-generate-plots 显式关闭。详情: {error}"
                )
    print("\nAttack completed")
    print(
        f"Direct TP ASR: {result.get('direct_tp_asr', 'n/a')} "
        f"({result.get('direct_native_tp_escaped', 'n/a')}/"
        f"{result.get('attacked_initial_tp_count', 'n/a')}); "
        f"Direct alert evasion rate: {result.get('target_evasion_rate', 'n/a')}"
    )
    print("Paper primary metrics:")
    print(f"  DirectTP-ASR: {result.get('direct_tp_asr', 'n/a')}")
    print(f"  EvasionAmplification: {result.get('evasion_amplification', 'n/a')}")
    print(f"  GraphModifyRate: {result.get('graph_modification_rate', 'n/a')}")
    print(f"  AverageAttackBudget: {result.get('average_attack_budget', 'n/a')}")
    print(
        f"Cost: detector_queries={result.get('total_detector_queries', 'n/a')}, "
        f"runtime_minutes={result.get('attack_elapsed_minutes', 'n/a')}"
    )
    if run_dir is not None:
        print(f"Experiment artifacts: {run_dir}")


if __name__ == "__main__":
    main()
