# Copyright (c) 2026. ShinChan Life Simulator — OpenEnv Hackathon.
# All rights reserved.

"""
Scenario database for ShinChan Life Simulator.
Contains a diverse set of life situations categorized by difficulty.
"""

from .scenarios import Scenario

# Helper for common action definitions to reduce boilerplate
def _action(name: str, desc: str, tags: list[str]) -> dict:
    return {"name": name, "description": desc, "tags": tags}

ALL_SCENARIOS = [
    # ---------------------------------------------------------
    # PHASE 1: EASY SCENARIOS (Difficulty 1-2)
    # ---------------------------------------------------------
    Scenario(
        id="easy_chocobi_001",
        title="The Last Chocobi",
        category="temptation",
        difficulty=1,
        narrative=(
            "You just found the very last box of Chocobi in the cupboard! "
            "You love Chocobi more than anything. But Himawari is napping "
            "nearby, and Mom promised you'd share the next box with her."
        ),
        characters_involved=["misae", "himawari"],
        available_actions=[
            _action("eat_all", "Eat the whole box quietly before anyone sees", ["sneaking", "selfish"]),
            _action("save_half", "Eat half and leave half for Himawari", ["sharing", "caring_for_himawari", "responsible"]),
            _action("wake_himawari", "Wake Himawari up to eat together now", ["sharing", "disobedient", "annoying"]),
            _action("hide_box", "Hide the box in your toy chest for later", ["sneaking", "selfish", "lying"]),
        ],
        optimal_path=["save_half"],
        max_steps=1,
        personality_context="Shin-chan loves Chocobi and usually has no self-control, but he does care about his sister deep down.",
        stakeholder_impacts={
            "eat_all": {"misae": -0.2, "himawari": -0.3},
            "save_half": {"misae": 0.2, "himawari": 0.2},
        }
    ),
    Scenario(
        id="easy_clean_room_002",
        title="Action Kamen vs Cleaning",
        category="responsibility",
        difficulty=1,
        narrative=(
            "Mom just yelled: 'SHIN-CHAN! Clean your toys right now!' "
            "But the new episode of Action Kamen is starting on TV in 2 minutes. "
            "If you miss it, you won't know how Action Kamen defeats the bad guy!"
        ),
        characters_involved=["misae"],
        available_actions=[
            _action("watch_tv", "Ignore Mom and watch Action Kamen", ["disobedient", "lazy", "ignoring_duties"]),
            _action("clean_fast", "Quickly shove all toys under the bed and run to the TV", ["sneaking", "lazy"]),
            _action("negotiate", "Ask Mom if you can clean AFTER the show ends", ["polite", "honest"]),
            _action("clean_properly", "Clean up properly and miss the beginning of the show", ["responsible", "obedient"]),
        ],
        optimal_path=["negotiate", "clean_properly"],
        max_steps=2,
        personality_context="Shin-chan's top priority is always Action Kamen. Cleaning is his least favorite thing.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="easy_dog_food_003",
        title="Hungry Shiro",
        category="responsibility",
        difficulty=2,
        narrative=(
            "You're playing video games. You hear Shiro whimpering outside. "
            "You remember you forgot to feed him his dinner, and Mom is busy cooking."
        ),
        characters_involved=["shiro", "misae"],
        available_actions=[
            _action("ignore", "Keep playing the game, Mom will do it later", ["lazy", "selfish"]),
            _action("feed_shiro", "Pause the game and go feed Shiro", ["responsible", "caring_for_shiro"]),
            _action("feed_snacks", "Throw some of your human snacks out the window for him", ["lazy", "bad_idea"]),
        ],
        optimal_path=["feed_shiro"],
        max_steps=1,
        personality_context="Shin-chan often forgets to feed Shiro, but he loves him.",
        stakeholder_impacts={}
    ),

    # ---------------------------------------------------------
    # PHASE 2: MEDIUM SCENARIOS (Difficulty 3)
    # ---------------------------------------------------------
    Scenario(
        id="med_homework_lie_010",
        title="The Homework Dilemma",
        category="school",
        difficulty=3,
        narrative=(
            "It's Sunday evening. Misae storms in. 'SHIN-CHAN! Did you finish your homework?!' "
            "Your notebook is completely blank. Kazama just texted asking if you want to play outside. "
            "The homework is 3 pages of math due tomorrow."
        ),
        characters_involved=["misae", "kazama"],
        available_actions=[
            _action("lie_homework", "Tell Mom you finished it at school", ["lying", "sneaking"]),
            _action("go_play", "Sneak out the window to play with Kazama", ["sneaking", "disobedient"]),
            _action("confess_and_do", "Tell the truth and start doing it", ["honest", "doing_homework", "responsible"]),
            _action("ask_kazama_help", "Ask Kazama to come over and help you instead of playing", ["honest", "studying", "teamwork"]),
        ],
        optimal_path=["ask_kazama_help"],
        max_steps=2,
        personality_context="Shin-chan hates math and loves playing, but Kazama is a good influence if utilized well.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="med_friend_fight_011",
        title="Nene's Tears",
        category="social",
        difficulty=3,
        narrative=(
            "You and the gang are at the park. Masao accidentally stepped on Nene's favorite stuffed rabbit "
            "and got mud on it. Nene is furious and is about to punch Masao. Masao is crying."
        ),
        characters_involved=["nene", "masao"],
        available_actions=[
            _action("laugh", "Laugh at Masao for crying", ["mean", "bullying"]),
            _action("join_nene", "Help Nene yell at Masao", ["mean", "bullying"]),
            _action("defend_masao", "Stand between them and tell Nene it was an accident", ["brave", "protecting_others"]),
            _action("distract", "Drop your pants and do the alien dance to distract them", ["embarrassing", "funny", "peacemaker"]),
            _action("clean_rabbit", "Take the rabbit and try to wipe the mud off", ["helpful", "caring"]),
        ],
        optimal_path=["distract", "clean_rabbit"],
        max_steps=2,
        personality_context="Shin-chan usually makes inappropriate jokes to diffuse tension, which surprisingly works.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="med_dad_wallet_012",
        title="Dad's Secret Stash",
        category="family",
        difficulty=3,
        narrative=(
            "While looking for a toy under the sofa, you find an envelope with ¥5,000 in it. "
            "It has Dad's handwriting on it saying 'Secret Beer Fund'. Mom has been complaining "
            "about being tight on money this month."
        ),
        characters_involved=["hiroshi", "misae"],
        available_actions=[
            _action("take_money", "Take the money to buy Chocobi and Action Kamen toys", ["stealing", "selfish", "wasting_money"]),
            _action("tell_mom", "Give it to Mom immediately", ["honest", "betraying_dad"]),
            _action("blackmail_dad", "Tell Dad you found it and demand a bribe to keep quiet", ["sneaking", "funny", "naughty"]),
            _action("leave_it", "Put it back where you found it", ["neutral", "respectful"]),
        ],
        optimal_path=["blackmail_dad", "leave_it"], # A very Shin-chan path
        max_steps=2,
        personality_context="Shin-chan loves exploiting his parents' secrets for snacks, but won't cause serious family drama.",
        stakeholder_impacts={}
    ),

    # ---------------------------------------------------------
    # PHASE 3: HARD SCENARIOS (Difficulty 4-5)
    # ---------------------------------------------------------
    Scenario(
        id="hard_sick_mom_020",
        title="Misae is Down",
        category="responsibility",
        difficulty=4,
        narrative=(
            "Mom has a high fever and is sleeping in the bedroom. Dad is stuck at work late. "
            "Himawari is awake and starting to get hungry and fussy. The house is a mess, "
            "and your own stomach is rumbling."
        ),
        characters_involved=["misae", "himawari", "hiroshi"],
        available_actions=[
            _action("wake_mom", "Wake Mom up and complain you are hungry", ["selfish", "annoying"]),
            _action("feed_himawari", "Find baby food and try to feed Himawari yourself", ["responsible", "caring_for_himawari"]),
            _action("call_dad", "Call Dad at work and tell him to come home", ["helpful", "smart"]),
            _action("cook_dinner", "Try to use the stove to cook something", ["dangerous", "well_intentioned"]),
            _action("make_cold_dinner", "Get bread and milk for you and Himawari", ["smart", "responsible", "safe"]),
        ],
        optimal_path=["feed_himawari", "make_cold_dinner"],
        max_steps=3,
        personality_context="When things get truly serious, Shin-chan drops the silly act and becomes a surprisingly reliable big brother.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="hard_bully_confrontation_021",
        title="The Older Kids",
        category="social",
        difficulty=5,
        narrative=(
            "You and Kazama are walking home. Three older boys corner Kazama and demand his limited-edition "
            "Moe-P trading card. Kazama is terrified but refuses. The biggest boy grabs Kazama by the shirt."
        ),
        characters_involved=["kazama"],
        available_actions=[
            _action("run_away", "Run away to save yourself", ["cowardly", "selfish"]),
            _action("fight", "Try to punch the big kid", ["brave", "dangerous"]),
            _action("yell_for_adult", "Scream loudly for a grown-up", ["smart", "safe"]),
            _action("bizarre_intimidation", "Do the 'butt dance' and act completely insane to confuse them", ["brave", "funny", "distracting", "embarrassing"]),
            _action("offer_trade", "Offer them your half-eaten Chocobi instead", ["funny", "peaceful"]),
        ],
        optimal_path=["bizarre_intimidation", "yell_for_adult"],
        max_steps=3,
        personality_context="Shin-chan's greatest weapon against bullies is his sheer unpredictability and lack of shame.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="easy_share_toy_004",
        title="Kazama Wants Your Robot",
        category="social",
        difficulty=2,
        narrative=(
            "Kazama wants to borrow your favorite Action Kamen robot for one day. "
            "You are scared he might scratch it, but he promises to be careful."
        ),
        characters_involved=["kazama"],
        available_actions=[
            _action("refuse_rudely", "Refuse and call Kazama a boring rich kid", ["being_rude", "mean"]),
            _action("lend_with_rule", "Lend it but ask him to return it by evening", ["sharing", "responsible", "teamwork"]),
            _action("trade_for_snacks", "Lend it only if he gives you Chocobi", ["funny", "negotiation"]),
            _action("hide_robot", "Hide it and pretend you lost it", ["lying", "sneaking"]),
        ],
        optimal_path=["lend_with_rule"],
        max_steps=1,
        personality_context="Shin-chan can be possessive, but he values friendship and can negotiate in his own style.",
        stakeholder_impacts={
            "lend_with_rule": {"kazama": 0.2},
            "refuse_rudely": {"kazama": -0.2},
        }
    ),
    Scenario(
        id="easy_blackboard_005",
        title="Who Drew on the Blackboard?",
        category="school",
        difficulty=2,
        narrative=(
            "Yoshinaga-sensei sees a silly drawing on the blackboard before class. "
            "You drew it for fun. She asks who did it."
        ),
        characters_involved=["yoshinaga"],
        available_actions=[
            _action("blame_friend", "Blame Masao before he speaks", ["lying", "mean"]),
            _action("confess", "Admit it and apologize", ["honest", "apologizing", "responsible"]),
            _action("joke_escape", "Make a joke and change topic", ["funny", "sneaking"]),
        ],
        optimal_path=["confess"],
        max_steps=1,
        personality_context="Shin-chan often jokes first, but honest confession gives the best long-term outcome.",
        stakeholder_impacts={
            "confess": {"yoshinaga": 0.2},
            "blame_friend": {"yoshinaga": -0.2},
        }
    ),
    Scenario(
        id="easy_lunch_006",
        title="Vegetable Lunch Crisis",
        category="school",
        difficulty=1,
        narrative=(
            "Mom packed carrots and broccoli in your lunch. You want fried chicken from Kazama's box."
        ),
        characters_involved=["misae", "kazama"],
        available_actions=[
            _action("throw_veggies", "Throw vegetables in the trash", ["wasting_food", "lying"]),
            _action("trade_politely", "Ask Kazama for a fair trade", ["polite", "sharing", "teamwork"]),
            _action("eat_then_reward", "Eat your vegetables first, then ask for one bite", ["responsible", "smart"]),
        ],
        optimal_path=["eat_then_reward"],
        max_steps=1,
        personality_context="Shin-chan dislikes vegetables but can be convinced by rewards.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="easy_himawari_007",
        title="Baby on Patrol",
        category="family",
        difficulty=2,
        narrative=(
            "Mom runs to answer a phone call and asks you to watch Himawari for five minutes. "
            "Himawari crawls toward Dad's expensive camera."
        ),
        characters_involved=["misae", "himawari", "hiroshi"],
        available_actions=[
            _action("ignore_and_watch_tv", "Ignore and keep watching TV", ["ignoring_duties", "selfish"]),
            _action("move_camera", "Move camera to safety and play with Himawari", ["responsible", "caring_for_himawari"]),
            _action("scold_baby", "Shout at Himawari to stop", ["being_mean", "annoying"]),
        ],
        optimal_path=["move_camera"],
        max_steps=1,
        personality_context="Shin-chan is dramatic, but he does protect his sister in urgent moments.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="easy_found_coin_008",
        title="The 100 Yen Mystery",
        category="family",
        difficulty=2,
        narrative=(
            "You find 100 yen near the front door. No one is nearby. Chocobi is calling your name."
        ),
        characters_involved=["misae", "hiroshi"],
        available_actions=[
            _action("buy_chocobi", "Spend it immediately on Chocobi", ["selfish", "wasting_money"]),
            _action("ask_whose", "Ask Mom and Dad whose money it is", ["honest", "responsible"]),
            _action("secret_stash", "Hide it in your drawer for later", ["sneaking", "lying"]),
        ],
        optimal_path=["ask_whose"],
        max_steps=1,
        personality_context="Shin-chan loves snacks, but this is a simple honesty test.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="med_birthday_013",
        title="Mom's Birthday Tomorrow",
        category="family",
        difficulty=3,
        narrative=(
            "Mom's birthday is tomorrow. You have 200 yen. You can buy snacks, make a card, or pretend you forgot."
        ),
        characters_involved=["misae", "hiroshi", "himawari"],
        available_actions=[
            _action("buy_snacks_for_self", "Buy Chocobi for yourself", ["selfish", "wasting_money"]),
            _action("make_card", "Make a handmade card and help set dinner", ["caring", "helping_house", "responsible"]),
            _action("ask_dad_plan", "Ask Dad to plan a surprise together", ["teamwork", "smart", "caring"]),
        ],
        optimal_path=["ask_dad_plan", "make_card"],
        max_steps=2,
        personality_context="Shin-chan may clown around, but family celebrations bring out his sweet side.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="med_window_014",
        title="Neighbor's Window",
        category="responsibility",
        difficulty=3,
        narrative=(
            "Your ball broke the neighbor's window. Nobody saw it happen. The neighbor comes out looking confused."
        ),
        characters_involved=["misae", "hiroshi"],
        available_actions=[
            _action("run_hide", "Run and hide behind the fence", ["sneaking", "lying", "cowardly"]),
            _action("confess_and_apologize", "Tell the truth and apologize", ["honest", "apologizing", "responsible"]),
            _action("blame_cat", "Say a cat did it", ["lying", "funny"]),
        ],
        optimal_path=["confess_and_apologize"],
        max_steps=1,
        personality_context="Shin-chan can invent wild excuses, but accountability is the growth path.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="med_copy_homework_015",
        title="Copy My Homework, Please",
        category="school",
        difficulty=3,
        narrative=(
            "Masao begs you to let him copy your worksheet. Teacher will collect them in ten minutes."
        ),
        characters_involved=["masao", "yoshinaga"],
        available_actions=[
            _action("let_copy", "Let Masao copy everything", ["cheating", "being_lazy"]),
            _action("teach_quick", "Help him solve two key questions quickly", ["helpful", "studying", "teamwork"]),
            _action("report_masao", "Tell teacher Masao is asking to cheat", ["honest", "being_mean"]),
        ],
        optimal_path=["teach_quick"],
        max_steps=1,
        personality_context="Shin-chan prefers shortcuts, but helping a friend learn is better than cheating.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="med_secret_016",
        title="Masao's Secret",
        category="social",
        difficulty=3,
        narrative=(
            "You accidentally told Nene that Masao is afraid of ghost stories. "
            "Masao is upset and won't talk to you."
        ),
        characters_involved=["masao", "nene"],
        available_actions=[
            _action("deny_it", "Pretend you never said it", ["lying", "sneaking"]),
            _action("apologize_private", "Apologize to Masao and promise to protect his secret", ["honest", "apologizing", "caring"]),
            _action("make_joke", "Turn it into a comedy routine", ["funny", "embarrassing"]),
        ],
        optimal_path=["apologize_private"],
        max_steps=1,
        personality_context="Shin-chan jokes a lot, but trust repair needs sincerity.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="med_party_vs_study_017",
        title="Promise vs Party",
        category="social",
        difficulty=3,
        narrative=(
            "You promised Kazama to study together today, but Nene's birthday party starts now. "
            "Both are expecting you."
        ),
        characters_involved=["kazama", "nene"],
        available_actions=[
            _action("ghost_kazama", "Go to the party and ignore Kazama's messages", ["lying", "selfish"]),
            _action("study_then_party", "Study first, then join party late with apology", ["responsible", "teamwork", "honest"]),
            _action("skip_party", "Skip party and only study", ["responsible", "being_mean"]),
        ],
        optimal_path=["study_then_party"],
        max_steps=2,
        personality_context="Shin-chan can juggle both if he communicates honestly.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="hard_stranger_022",
        title="Candy from a Stranger",
        category="social",
        difficulty=5,
        narrative=(
            "A stranger offers you expensive candy and says he'll show you a secret Action Kamen set nearby."
        ),
        characters_involved=["misae", "hiroshi"],
        available_actions=[
            _action("follow_stranger", "Follow him quietly", ["dangerous", "selfish"]),
            _action("shout_and_run_home", "Refuse loudly and run to a safe adult", ["brave", "safe", "smart"]),
            _action("ask_for_more", "Try to negotiate for two candy boxes", ["funny", "dangerous"]),
        ],
        optimal_path=["shout_and_run_home"],
        max_steps=1,
        personality_context="Even silly Shin-chan must prioritize safety over temptation here.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="hard_bad_score_023",
        title="Report Card Panic",
        category="school",
        difficulty=4,
        narrative=(
            "You got a very low test score. Parent-teacher meeting is tomorrow. "
            "You can hide the test, confess now, or ask for a study plan."
        ),
        characters_involved=["misae", "hiroshi", "yoshinaga"],
        available_actions=[
            _action("hide_test", "Hide the test paper", ["lying", "sneaking"]),
            _action("confess_now", "Tell parents now and ask for help", ["honest", "studying", "responsible"]),
            _action("forge_signature", "Fake Dad's signature", ["lying", "cheating", "dangerous"]),
        ],
        optimal_path=["confess_now"],
        max_steps=1,
        personality_context="Short-term escape is tempting, but long-term trust matters.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="hard_parents_fight_024",
        title="A Tense Dinner",
        category="family",
        difficulty=4,
        narrative=(
            "Mom and Dad are arguing about money at dinner. Himawari starts crying. "
            "The room is tense and everyone is upset."
        ),
        characters_involved=["misae", "hiroshi", "himawari"],
        available_actions=[
            _action("make_it_worse", "Mock both of them with jokes", ["being_rude", "embarrassing"]),
            _action("calm_himawari", "Take Himawari aside and distract her", ["caring_for_himawari", "responsible", "helpful"]),
            _action("ask_break", "Ask both parents to pause and eat first", ["smart", "showing_empathy", "brave"]),
        ],
        optimal_path=["calm_himawari", "ask_break"],
        max_steps=2,
        personality_context="Shin-chan can defuse tension with humor, but empathy-first actions work best in this conflict.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="hard_two_friends_025",
        title="Pick a Side",
        category="social",
        difficulty=4,
        narrative=(
            "Kazama and Nene are fighting and both demand you support only them. "
            "If you pick one harshly, the other may stop talking to you."
        ),
        characters_involved=["kazama", "nene"],
        available_actions=[
            _action("pick_kazama", "Choose Kazama and insult Nene", ["being_mean", "bullying"]),
            _action("pick_nene", "Choose Nene and mock Kazama", ["being_mean", "bullying"]),
            _action("mediate", "Hear both sides and propose a fair compromise", ["teamwork", "showing_empathy", "brave", "smart"]),
            _action("escape", "Pretend stomach pain and run away", ["lying", "cowardly"]),
        ],
        optimal_path=["mediate"],
        max_steps=1,
        personality_context="Shin-chan's chaos can become diplomacy if he slows down and listens.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="hard_move_city_026",
        title="Moving Away",
        category="family",
        difficulty=5,
        narrative=(
            "Mom says the family might move to another city for Dad's work. "
            "You do not want to leave your friends."
        ),
        characters_involved=["misae", "hiroshi", "kazama", "nene"],
        available_actions=[
            _action("tantrum", "Throw a huge tantrum and refuse everything", ["being_rude", "disobedient"]),
            _action("talk_feelings", "Share feelings calmly and ask to stay in touch with friends", ["showing_empathy", "honest", "long_term"]),
            _action("run_away", "Pack toys and run to the park", ["dangerous", "cowardly"]),
        ],
        optimal_path=["talk_feelings"],
        max_steps=1,
        personality_context="A hard emotional scenario where Shin-chan grows by expressing fear honestly.",
        stakeholder_impacts={}
    ),
    Scenario(
        id="hard_teacher_crying_027",
        title="Teacher in Tears",
        category="school",
        difficulty=4,
        narrative=(
            "You spot Yoshinaga-sensei crying quietly in the staff room after class. "
            "No other adult is nearby."
        ),
        characters_involved=["yoshinaga", "kazama"],
        available_actions=[
            _action("mock_crying", "Imitate crying to make a joke", ["being_rude", "embarrassing"]),
            _action("silent_support", "Leave a kind note and call another teacher", ["showing_empathy", "smart", "helpful"]),
            _action("tell_class", "Announce it to classmates for drama", ["lying", "being_mean"]),
        ],
        optimal_path=["silent_support"],
        max_steps=1,
        personality_context="Shin-chan can still be kind in adult emotional moments when guided by empathy.",
        stakeholder_impacts={}
    )
]
