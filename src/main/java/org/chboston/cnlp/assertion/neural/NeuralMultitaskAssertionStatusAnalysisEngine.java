package org.chboston.cnlp.assertion.neural;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.ctakes.neural.feature.TokenSequenceWithConstrainedWindowExtractor;
import org.apache.ctakes.typesystem.type.textsem.EntityMention;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.uima.UIMAFramework;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.Logger;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;

public class NeuralMultitaskAssertionStatusAnalysisEngine extends
    CleartkAnnotator<String> {

  Logger logger = UIMAFramework.getLogger(NeuralMultitaskAssertionStatusAnalysisEngine.class);

  private final static int WINDOW_SIZE = 10;
//  private TokensSequenceWithWindowExtractor seqExtractor = new TokensSequenceWithWindowExtractor(WINDOW_SIZE);
  private TokenSequenceWithConstrainedWindowExtractor<IdentifiedAnnotation, Sentence> seqExtractor = new TokenSequenceWithConstrainedWindowExtractor<>(WINDOW_SIZE);
    
  @Override
  public void process(JCas jCas) throws AnalysisEngineProcessException {
    for(Sentence sent : JCasUtil.select(jCas, Sentence.class)){
      Collection<IdentifiedAnnotation> entities = JCasUtil.selectCovered(jCas, IdentifiedAnnotation.class, sent);
      for (IdentifiedAnnotation identifiedAnnotation : entities)
      {
        if (!(identifiedAnnotation instanceof EntityMention || identifiedAnnotation instanceof EventMention))
        {
          continue;
        }
        IdentifiedAnnotation target = identifiedAnnotation;

        List<Feature> feats = new ArrayList<>();

        feats.addAll(this.seqExtractor.extract(jCas, target, sent));
//        feats.addAll(this.seqExtractor.extract(jCas, target));
        Instance<String> instance = new Instance<>(feats);

        if(this.isTraining()){
          int polarity = target.getPolarity();
          int uncertainty = target.getUncertainty();
          boolean generic = target.getGeneric();
          String subject = target.getSubject();
          int history = target.getHistoryOf();
          boolean conditional = target.getConditional();

          String label = String.format("polarity=%d uncertainty=%d generic=%s subject=%s history=%d conditional=%s", polarity, uncertainty, generic, subject, history, conditional);
          instance.setOutcome(label);
          this.dataWriter.write(new Instance<>(label, feats));
        }else{
          String label = this.classifier.classify(feats);
          String[] labels = label.split(" ");
          int polarity = Integer.valueOf(labels[0].split("=")[1]);
          int uncertainty = Integer.valueOf(labels[1].split("=")[1]);
          boolean generic = Boolean.valueOf(labels[2].split("=")[1]);
          String subject = labels[3].split("=")[1];
          int history = Integer.parseInt(labels[4].split("=")[1]);
          boolean conditional = Boolean.valueOf(labels[5].split("=")[1]);

          target.setPolarity(polarity);
          target.setUncertainty(uncertainty);
          target.setGeneric(generic);
          target.setSubject(subject);
          target.setHistoryOf(history);
          target.setConditional(conditional);
        }
      }
    }
  }  
}
