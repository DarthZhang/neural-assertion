package org.chboston.cnlp.assertion.neural;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.ctakes.core.cleartk.ContinuousTextExtractor;
import org.apache.ctakes.core.cleartk.ContinuousTextExtractor.OovStrategy;
import org.apache.ctakes.core.cleartk.FollowingWithPadding;
import org.apache.ctakes.core.cleartk.PrecedingWithPadding;
import org.apache.ctakes.core.cleartk.SumContext;
import org.apache.ctakes.core.util.DocumentIDAnnotationUtil;
import org.apache.ctakes.typesystem.type.syntax.BaseToken;
import org.apache.ctakes.typesystem.type.textsem.EntityMention;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.uima.UIMAFramework;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.Level;
import org.apache.uima.util.Logger;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;
import org.cleartk.ml.feature.extractor.CleartkExtractor;
import org.cleartk.ml.feature.extractor.CleartkExtractor.Bag;
import org.cleartk.ml.feature.extractor.CleartkExtractor.Covered;
import org.cleartk.ml.feature.extractor.CleartkExtractorException;

public class NeuralMultitaskAssertionStatusAnalysisEngine extends
    CleartkAnnotator<String> {

  Logger logger = UIMAFramework.getLogger(NeuralMultitaskAssertionStatusAnalysisEngine.class);
  private ContinuousTextExtractor continuousText;
  protected CleartkExtractor<IdentifiedAnnotation,BaseToken> tokenVectorContext;
  final String vectorFile = "org/chboston/cnlp/assertion/neural/mimic_vectors.txt";

  @Override
  public void initialize(UimaContext context)
      throws ResourceInitializationException {
    super.initialize(context);
    try {
      this.continuousText = new ContinuousTextExtractor(vectorFile, OovStrategy.EMPTY_VECTOR);
      this.tokenVectorContext = new CleartkExtractor<>(
          BaseToken.class,
          continuousText,
          new PrecedingWithPadding(5, this.continuousText.getEmbeddingsDimensionality()),
          new SumContext(new Bag(new Covered())),
          new FollowingWithPadding(5, this.continuousText.getEmbeddingsDimensionality())
//          new SumContext(new Bag(new Preceding(5))),
//          new MaxContext(new Bag(new Preceding(5))),
//          new MinContext(new Bag(new Preceding(5))),
//          new Covered());
//          new MinContext(new Bag(new Following(5))),
//          new MaxContext(new Bag(new Following(5))),
//          new SumContext(new Bag(new Following(5))));
          );
    } catch (CleartkExtractorException e) {
      System.err.println("cannot find file: "+ vectorFile);
      e.printStackTrace();
    }
  }
  
  @Override
  public void process(JCas jCas) throws AnalysisEngineProcessException {
    String documentId = DocumentIDAnnotationUtil.getDocumentID(jCas);
    if(documentId != null){
//      logger.log(Level.INFO, "Processing document " + documentId);
    }
    
    Collection<IdentifiedAnnotation> entities = JCasUtil.select(jCas, IdentifiedAnnotation.class);
    for (IdentifiedAnnotation identifiedAnnotation : entities)
    {
      if (!(identifiedAnnotation instanceof EntityMention || identifiedAnnotation instanceof EventMention))
      {
        continue;
      }
      IdentifiedAnnotation target = identifiedAnnotation;

      List<Feature> feats = new ArrayList<>();

      feats.addAll(this.tokenVectorContext.extract(jCas, target));
      Instance<String> instance = new Instance<>(feats);
      
      if(this.isTraining()){
        int polarity = target.getPolarity();
        int uncertainty = target.getUncertainty();
        boolean generic = target.getGeneric();
        String subject = target.getSubject();
        int history = target.getHistoryOf();
        boolean conditional = target.getConditional();

        String label = String.format("polarity=%d#uncertainty=%d#generic=%s#subject=%s#history=%d#conditional=%s", polarity, uncertainty, generic, subject, history, conditional);
        instance.setOutcome(label);
        this.dataWriter.write(new Instance<>(label, feats));
      }else{
        String label = this.classifier.classify(feats);
//        logger.info(label);
        String[] labels = label.split("#");
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
