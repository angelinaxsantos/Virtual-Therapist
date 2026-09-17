#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates the training dataset for the safety layer (crisis/suicide/self-harm).

Design decisions (agreed on throughout the project):
- English only: fine-tuning is done only in EN; PT and other languages are
  obtained by translating the final response, not through bilingual training data.
- Adapted, not copied, from the RACGP/GPMHSC and WHO LIVE LIFE guidelines.
- No detailed lethality-assessment questions (see methodology note).
- The phone number is NEVER written as fixed text - the {{CRISIS_LINE}}
  placeholder is used instead, replaced later by code (crisis_lookup.py) with
  the correct number for the user's country. The model never "memorizes" a number.
- Each response combines several phases (acknowledge, invite, validate, refer,
  close) into one fluid message, not as separate turns.
- Varied combinations (not the same phrase-response pair always), to avoid
  excessive repetition of fixed phrases.
- Includes multi-turn examples, where risk is revealed gradually over the
  conversation, not only through a single isolated trigger.
"""
import json
import random

random.seed(42)

# ---------------------------------------------------------------------------
# PART 1 - Detection phrases (by category), English only
# ---------------------------------------------------------------------------
CATEGORIES = {
    "A_hopelessness": [
        "I can't see a way out of this.",
        "There's no point in trying anymore.",
        "I just can't take this anymore.",
        "Nothing is ever going to get better.",
        "I've lost all hope.",
        "What's even the point of going on?",
        "This pain is never going to end.",
        "I feel like I'm never going to feel okay again.",
        "Everything just feels hopeless right now.",
        "I don't see any way things could improve.",
        "I'm so tired of hoping things will change.",
        "Nothing I do matters anymore.",
        "I feel completely stuck with no way out.",
        "I keep thinking things will never get better.",
        "I've given up on things improving.",
    ],
    "B_burden_isolation": [
        "Everyone would be better off without me.",
        "No one really understands me.",
        "I feel completely alone in this.",
        "I'm such a burden to everyone around me.",
        "Nobody would even miss me.",
        "I feel like I'm just in everyone's way.",
        "Please just leave me alone, it's easier for everyone.",
        "My family would be relieved if I wasn't here.",
        "I don't think anyone would notice if I disappeared.",
        "I feel like a weight on the people I love.",
        "I'm tired of being a problem for everyone.",
        "It feels like nobody would care either way.",
        "I isolate myself because I think I make things worse.",
        "People are better off staying away from me.",
        "I feel invisible, like I don't matter to anyone.",
    ],
    "C_farewell": [
        "I want to get all my affairs in order.",
        "I keep thinking about what happens after I'm gone.",
        "I want you to have this, just in case.",
        "Thank you for everything, I mean it.",
        "I've been giving away some of my things lately.",
        "I wrote some letters to people, just to say goodbye.",
        "I want to make sure everything is settled before anything happens.",
        "I've been thinking about how I'd want to be remembered.",
        "I just wanted to say I really appreciate you, in case I don't get another chance.",
        "I've been sorting through my stuff, deciding who gets what.",
        "If something happened to me, I'd want you to know I loved you.",
        "I feel like I'm tying up loose ends.",
        "I keep saying goodbye to people without really explaining why.",
        "I've made peace with how things might end.",
        "I want to leave things in order for my family.",
    ],
    "D_direct_ideation": [
        "Sometimes I think about ending it all.",
        "I don't want to be alive anymore.",
        "I think about dying a lot lately.",
        "I've thought about suicide.",
        "I have thoughts of killing myself.",
        "I just want to disappear forever.",
        "What if I just didn't wake up tomorrow?",
        "I've been thinking about hurting myself badly.",
        "Sometimes I wish I could just not exist.",
        "I keep having thoughts about ending my life.",
        "I don't think I want to keep living.",
        "I've thought about how I might do it.",
        "I feel like ending everything is the only way out.",
        "I can't stop thinking about not being here anymore.",
        "I've been planning what I'd do to end things.",
    ],
    "E_behavior_change": [
        "I've stopped going out at all.",
        "I quit doing things I used to enjoy.",
        "I've been way more irritable and withdrawn lately.",
        "I haven't slept properly in days.",
        "I barely eat anything anymore.",
        "I can't focus on anything these days.",
        "I've pulled away from all my friends.",
        "I stopped answering messages from people who care about me.",
        "I don't leave my room much anymore.",
        "I've missed a lot of work and school lately.",
        "I've completely lost interest in everything.",
        "I've been sleeping way too much lately.",
        "I feel numb most of the time now.",
        "I've stopped taking care of myself.",
        "I don't talk to anyone anymore.",
    ],
    "F_self_harm_risk": [
        "I've been hurting myself.",
        "I don't care if something bad happens to me.",
        "I've been drinking a lot more than usual.",
        "I don't care if I get into an accident.",
        "I've started cutting again.",
        "I've been taking risks I normally wouldn't.",
        "I've been using more than I should lately.",
        "I feel like hurting myself when things get bad.",
        "I've thought about stopping my medication on purpose.",
        "Sometimes I hurt myself to feel something.",
        "I've been driving recklessly, not really caring what happens.",
        "I've had thoughts of hurting myself again.",
        "I don't take care of my safety like I used to.",
        "I've been isolating so I can hurt myself without anyone knowing.",
        "I feel like I deserve to be in pain.",
    ],
}

# ---------------------------------------------------------------------------
# PART 2 - Safe response phases, English only
# ---------------------------------------------------------------------------
PHASE_ACKNOWLEDGE = [
    "Thank you for telling me this. It's not easy to share something like that, and I'm really glad you did.",
    "I hear you, and I want you to know this matters. I'm here with you right now.",
    "I'm really grateful you trusted me with this. I'm not going to judge you for it.",
    "That sounds like an incredibly heavy thing to carry. Thank you for letting me in.",
]

PHASE_INVITE = [
    "Would you like to tell me more about what's been going on?",
    "How long have you been feeling this way?",
    "What's been the hardest part for you lately?",
    "Can you help me understand what's been happening?",
]

PHASE_VALIDATE = [
    "It makes complete sense that you'd feel this way given everything you're carrying. And you don't have to go through this alone.",
    "What you're feeling is real, and it matters. You deserve support, not just to push through this by yourself.",
    "I understand this might feel unbearable right now. That doesn't mean it has to stay this way.",
    "This sounds like a lot of pain to hold onto alone. You shouldn't have to.",
]

# {{CRISIS_LINE}} is replaced later by code, never generated by the model.
PHASE_REFER = [
    "I want you to know that real support is available right now — you can reach {{CRISIS_LINE}}, any time, for free and confidentially.",
    "There are people trained to help with exactly this, right now. Please consider reaching out to {{CRISIS_LINE}}.",
    "If at any point you feel like you can't keep yourself safe, please contact {{CRISIS_LINE}} or emergency services immediately.",
]

PHASE_CLOSE = [
    "I'm still here if you want to keep talking about this.",
    "You don't have to figure everything out right now. I'm here, at your pace.",
    "You can come back and talk to me about this any time you need to.",
]


def build_response(i):
    """Combines the phases independently and randomly (but reproducibly, fixed
    seed), to maximize distinct combinations instead of short repeated cycles."""
    ack = random.choice(PHASE_ACKNOWLEDGE)
    invite = random.choice(PHASE_INVITE)
    validate = random.choice(PHASE_VALIDATE)
    refer = random.choice(PHASE_REFER)
    close = random.choice(PHASE_CLOSE)
    structure = random.choice(["short", "medium", "long"])
    if structure == "short":
        return f"{ack} {validate} {refer}"
    elif structure == "medium":
        return f"{ack} {invite} {refer} {close}"
    else:
        return f"{ack} {validate} {invite} {refer} {close}"


# ---------------------------------------------------------------------------
# Generate single-turn pairs: 2 response combinations per detection phrase
# ---------------------------------------------------------------------------
single_turn_examples = []
counter = 0
for category, phrases in CATEGORIES.items():
    for phrase in phrases:
        for variant in range(2):  # 2 different responses per phrase
            response = build_response(counter)
            single_turn_examples.append({
                "messages": [
                    {"role": "user", "content": phrase},
                    {"role": "assistant", "content": response},
                ],
                "category": category,
                "type": "single_turn",
            })
            counter += 1

# ---------------------------------------------------------------------------
# Multi-turn examples: risk is revealed gradually, not only in a single
# isolated trigger message
# ---------------------------------------------------------------------------
multi_turn_examples = [
    [
        ("I've been really tired lately, I don't know why.",
         "That sounds draining. How long has this been going on?"),
        ("A few weeks now. I also don't sleep or eat much anymore, and honestly, sometimes I think it'd be easier to just not be here.",
         "Thank you for telling me that. It sounds like you're carrying a lot right now, and I want you to know that matters. "
         "I want you to know real support is available right now — you can reach {{CRISIS_LINE}}, any time, for free and confidentially. "
         "I'm still here if you want to keep talking about this."),
    ],
    [
        ("Work has been really overwhelming this month.",
         "That sounds like a lot to deal with. What's been the hardest part?"),
        ("Just feeling like nothing I do is good enough. I've started giving some of my things away to people, just in case.",
         "I hear you, and I want you to know this matters. When you say 'just in case', I want to check in with you directly — "
         "are you having thoughts of ending your life? Whatever the answer, please know that support is available right now at {{CRISIS_LINE}}, "
         "any time, for free and confidentially."),
        ("Yeah... sometimes I do think about it.",
         "Thank you for being honest with me, that takes courage. What you're feeling is real, and you don't have to go through it alone. "
         "Please reach out to {{CRISIS_LINE}} right now, or emergency services if you feel unsafe in this moment. I'm still here with you."),
    ],
    [
        ("I feel like everyone would just be fine without me around.",
         "I hear you, and that sounds like a really painful thing to be feeling. Can you tell me more about what's been happening?"),
        ("I don't know, I just feel like a burden all the time. I've been thinking about ending it more and more.",
         "Thank you for trusting me with this. What you're feeling is real, and it matters — you deserve support, not to carry this alone. "
         "Please reach out to {{CRISIS_LINE}} right now, they're trained to help with exactly this. I'm still here if you want to keep talking."),
    ],
    [
        ("I haven't left my room in days.",
         "That sounds really hard. What's been going on for you?"),
        ("I just don't see the point anymore. Nothing feels like it matters.",
         "That sounds like an incredibly heavy thing to carry. How long have you been feeling this way?"),
        ("A while now. Honestly I've thought about how I'd end things.",
         "I'm really glad you told me that. This sounds like a lot of pain to hold onto alone, and you shouldn't have to. "
         "Please contact {{CRISIS_LINE}} or emergency services right now if you feel unsafe. I'm not going anywhere."),
    ],
    [
        ("I keep having these thoughts I don't really want to have.",
         "I'm here to listen, whatever they are. Can you tell me a bit more?"),
        ("Thoughts about not wanting to be alive anymore. I haven't told anyone.",
         "Thank you for telling me, that took real courage. What you're feeling is real, and it matters. "
         "I want you to know that real support is available right now — you can reach {{CRISIS_LINE}}, any time, for free and confidentially."),
        ("I don't know if I could actually call anyone.",
         "That's okay, you don't have to do it alone or figure it all out right now. Even just staying here talking with me is a start. "
         "If at any point it feels unsafe, please reach out to {{CRISIS_LINE}} or emergency services. I'm still here with you."),
    ],
    [
        ("My sleep has been all over the place.",
         "That can really wear a person down. What's been on your mind?"),
        ("Honestly, I've been drinking a lot more than usual, and I don't really care what happens to me anymore.",
         "I hear you, and I want you to know this matters. It makes sense you'd feel this way given everything you're carrying, "
         "and you don't have to go through this alone. There's support available at {{CRISIS_LINE}}, any time, free and confidential."),
    ],
    [
        ("I wrote a letter to my sister yesterday, just to tell her I love her.",
         "That sounds like something meaningful to you. What made you want to write it now?"),
        ("I guess I've been thinking a lot about endings lately. Like maybe it would be better if I wasn't around.",
         "Thank you for sharing that with me. What you're feeling is real, and it matters — you deserve support, not to carry this by yourself. "
         "Please reach out to {{CRISIS_LINE}} right now, they're trained to help with exactly this."),
    ],
    [
        ("I don't really talk to anyone anymore.",
         "That sounds lonely. Has something changed recently?"),
        ("I just feel like a burden whenever I do. Sometimes I think everyone would be relieved if I was gone.",
         "I'm really glad you told me this. This sounds like a lot of pain to hold onto alone, and you shouldn't have to. "
         "Please contact {{CRISIS_LINE}} right now, or emergency services if you feel unsafe in this moment."),
        ("Okay. I don't know if I will, but okay.",
         "That's alright. Just knowing it's there, for whenever you're ready, is enough for now. I'm still here if you want to keep talking."),
    ],
    [
        ("Today has been a really bad day.",
         "I'm sorry to hear that. What happened?"),
        ("Nothing in particular. I just keep thinking that ending everything would be the only way out.",
         "Thank you for telling me that directly, it matters. Please reach out to {{CRISIS_LINE}} right now, or emergency services if you feel "
         "unsafe in this moment. I'm still here with you, and we can keep talking as long as you need."),
    ],
    [
        ("I've been cutting again.",
         "Thank you for telling me. I'm not going to judge you for it. How long has this been happening again?"),
        ("A few weeks. It's the only thing that makes the pain feel manageable, but honestly I've also thought about just ending it.",
         "That sounds incredibly hard to carry alone. What you're feeling is real, and it matters. Please reach out to {{CRISIS_LINE}} right now — "
         "they're trained to help with exactly this kind of pain."),
    ],
    [
        ("Can I tell you something I haven't told anyone?",
         "Of course. I'm here, and I'm listening."),
        ("I've been planning how I would end my life.",
         "Thank you for trusting me with something this important. I want you to know that real support is available right now — "
         "please contact {{CRISIS_LINE}} or emergency services immediately, especially given what you've just shared. I'm right here with you."),
    ],
    [
        ("I feel like I'm just going through the motions lately.",
         "That sounds exhausting. What does a typical day look like for you right now?"),
        ("Wake up, feel numb, go to bed. Sometimes I wonder what the point of any of it is anymore.",
         "That sounds like a heavy place to be in. How long have you been feeling this numbness?"),
        ("Months, honestly. I don't think I want to keep living like this.",
         "Thank you for being so honest with me. What you're feeling is real, and it matters — you deserve support, not to carry this alone. "
         "Please reach out to {{CRISIS_LINE}} right now, any time, free and confidential."),
    ],
]

multi_turn_formatted = []
for convo in multi_turn_examples:
    messages = []
    for user_msg, assistant_msg in convo:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    multi_turn_formatted.append({
        "messages": messages,
        "category": "multi_turn_gradual_risk",
        "type": "multi_turn",
    })

# ---------------------------------------------------------------------------
# Save everything as JSONL
# ---------------------------------------------------------------------------
all_examples = single_turn_examples + multi_turn_formatted
random.shuffle(all_examples)

out_path = "/mnt/user-data/outputs/crisis_safety_dataset.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Total: {len(all_examples)} examples "
      f"({len(single_turn_examples)} single-turn + {len(multi_turn_formatted)} multi-turn)")
print(f"Saved to: {out_path}")
