import pytest
from app.supervisor.agent import SupervisorAgent
from app.models.context import LiveTriggerType, ActionType

@pytest.mark.asyncio
async def test_live_check_grammar():
    # Initialize supervisor without heavy models (uses rule-based fallback)
    agent = SupervisorAgent(llm_client=None, use_llm_synthesis=False)
    
    # Test sentence check with grammar error
    content = "She have a car."
    trigger = LiveTriggerType.SENTENCE
    
    actions = await agent.live_check(content, trigger)
    
    assert len(actions) > 0
    assert actions[0].category == "grammar"
    assert "Subject-verb agreement" in actions[0].reasoning

@pytest.mark.asyncio
async def test_live_check_tone():
    agent = SupervisorAgent(llm_client=None)
    
    # Test pause with informal language
    # "wanna" and "stuff" -> 2 issues -> formality 0.4 < 0.6 threshold
    content = "I wanna do stuff."
    trigger = LiveTriggerType.PAUSE
    
    actions = await agent.live_check(content, trigger)
    
    # Should catch "wanna"
    found_issue = False
    for action in actions:
        if action.category == "tone" and "wanna" in action.suggestion:
            found_issue = True
            break
            
    assert found_issue

@pytest.mark.asyncio
async def test_live_check_paragraph_trigger():
    agent = SupervisorAgent(llm_client=None)
    
    # Paragraph trigger should invoke coherence (but we need enough text)
    content = "First sentence. Second sentence."
    trigger = LiveTriggerType.PARAGRAPH
    
    # Even if coherence returns nothing for short text, we ensure no crash
    actions = await agent.live_check(content, trigger)
    assert isinstance(actions, list)
