using System.Collections.Generic;
using Dissertation.Playback;
using NUnit.Framework;

public class SequenceTests
{
    const string SampleJson = @"{
      ""meta"": {""sequence_id"": ""S001"", ""policy"": ""scorer"",
                 ""world"": ""w.json"", ""seed"": 42, ""n_cycles"": 2},
      ""steps"": [
        {""cycle"": 1, ""location"": ""tavern"", ""action"": ""chat"", ""moved"": true,
         ""location_probs"": {""tavern"": 1.0}, ""action_probs"": {""chat"": 1.0}},
        {""cycle"": 2, ""location"": ""tavern"", ""action"": ""drink"", ""moved"": false}
      ]}";

    [Test]
    public void ParsesNeutralFieldsAndIgnoresArchiveFields()
    {
        var seq = Sequence.FromJson(SampleJson);
        Assert.AreEqual("S001", seq.meta.sequence_id);
        Assert.AreEqual(2, seq.steps.Count);
        Assert.AreEqual("chat", seq.steps[0].action);
        Assert.IsFalse(seq.steps[1].moved);
    }

    [Test]
    public void RejectsStepCountMismatch()
    {
        var bad = SampleJson.Replace(@"""n_cycles"": 2", @"""n_cycles"": 3");
        Assert.Throws<System.InvalidOperationException>(() => Sequence.FromJson(bad));
    }

    [Test]
    public void RejectsFirstStepWithoutMove()
    {
        var bad = SampleJson.Replace(
            @"""action"": ""chat"", ""moved"": true",
            @"""action"": ""chat"", ""moved"": false");
        Assert.Throws<System.InvalidOperationException>(() => Sequence.FromJson(bad));
    }

    [Test]
    public void RejectsInconsistentMovedFlag()
    {
        var bad = SampleJson.Replace(
            @"""action"": ""drink"", ""moved"": false",
            @"""action"": ""drink"", ""moved"": true");
        Assert.Throws<System.InvalidOperationException>(() => Sequence.FromJson(bad));
    }

    [Test]
    public void ModelAdvancesInOrderThenReturnsNull()
    {
        var seq = Sequence.FromJson(SampleJson);
        var model = new PlaybackModel(seq.steps);
        Assert.IsNull(model.Current);
        Assert.AreEqual(1, model.Advance().cycle);
        Assert.AreEqual(2, model.Advance().cycle);
        Assert.IsTrue(model.Finished);
        Assert.IsNull(model.Advance());
        Assert.AreEqual(2, model.Current.cycle);
    }

    [Test]
    public void DtoNeverDeclaresConditionFields()
    {
        // blinding constraint: if someone adds one of these to SequenceMeta,
        // JsonUtility would start parsing it and the UI could leak it
        var forbidden = new List<string>
            { "policy", "checkpoint", "personality_name", "ocean", "seed" };
        foreach (var name in forbidden)
            Assert.IsNull(typeof(SequenceMeta).GetField(name),
                $"condition field {name} must not be parsed by the player");
    }

    [Test]
    public void LayoutCoversAllWorldLocations()
    {
        // must stay in sync with data/world.json
        var expected = new List<string>
            { "tavern", "library", "chapel", "market", "forest", "arena", "enemy_camp" };
        foreach (var id in expected)
            Assert.IsTrue(LocationLayout.Entries.ContainsKey(id), $"missing layout: {id}");
        Assert.AreEqual(expected.Count, LocationLayout.Entries.Count);
    }
}
