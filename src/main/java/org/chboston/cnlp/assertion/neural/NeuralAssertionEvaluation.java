package org.chboston.cnlp.assertion.neural;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.ctakes.assertion.eval.AnnotationStatisticsCompact;
import org.apache.ctakes.assertion.eval.AssertionEvaluation;
import org.apache.ctakes.assertion.eval.AssertionEvaluation.ReferenceAnnotationsSystemAssertionClearer;
import org.apache.ctakes.assertion.eval.AssertionEvaluation.ReferenceIdentifiedAnnotationsSystemToGoldCopier;
import org.apache.ctakes.assertion.eval.XMIReader;
import org.apache.ctakes.typesystem.type.textsem.EntityMention;
import org.apache.ctakes.typesystem.type.textsem.EventMention;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.uima.UIMAFramework;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASException;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.factory.TypeSystemDescriptionFactory;
import org.apache.uima.fit.pipeline.JCasIterator;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.Level;
import org.apache.uima.util.Logger;
import org.cleartk.eval.Evaluation_ImplBase;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.jar.DefaultDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.ml.jar.JarClassifierBuilder;
import org.cleartk.ml.python.keras.KerasStringOutcomeDataWriter;
import org.cleartk.ml.script.ScriptStringOutcomeDataWriter;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class NeuralAssertionEvaluation extends Evaluation_ImplBase<File, Map<String, AnnotationStatisticsCompact<String>>> {
  public static class Options {
    @Option(
        name = "--train-dir",
        usage = "specify the directory containing the XMI training files (for example, /NLP/Corpus/Relations/mipacq/xmi/train)",
        required = true)
    public String trainDirectory;
    
    @Option(
        name = "--test-dir",
        usage = "specify the directory containing the XMI testing files (for example, /NLP/Corpus/Relations/mipacq/xmi/test)",
        required = true)
    public File testDirectory;
    
    @Option(
        name = "--skip-train",
        usage = "skip training and just evaluate on the specified test set",
        required = false)
    public boolean skipTrain=false;
    
    @Option(
        name = "--skip-write",
        usage = "Skip writing the data and just train and evaluate",
        required = false)
    public boolean skipWrite=false;
  }
  
  public NeuralAssertionEvaluation(File baseDirectory) {
    super(baseDirectory);
  }

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    CmdLineParser parser = new CmdLineParser(options);
    parser.parseArgument(args);

    NeuralAssertionEvaluation eval = new NeuralAssertionEvaluation(new File("target/models/neural/"));

    List<File> trainFiles = new ArrayList<>();
    if(options.skipTrain){
      eval.skipTrain = true;
    }else if(options.skipWrite){
      eval.skipWrite = true;
    }else{
      if (null != options.trainDirectory) {
        String[] dirs = options.trainDirectory.split("[;:]");
        for (String dir : dirs) {
          File trainDir = new File(dir);
          if (trainDir.listFiles()!=null) {
            for (File f : trainDir.listFiles()) {
              trainFiles.add(f);
            }
          }
        }
      }
    }

    List<File> testFiles = new ArrayList<>();
    testFiles = Arrays.asList(options.testDirectory.listFiles());

    // if this is a new run make sure the models directory is empty:
    if(!(options.skipTrain || options.skipWrite)){
      FileUtils.deleteDirectory(new File("target/models/neural/train_and_test/"));
    }
    
    Map<String, AnnotationStatisticsCompact<String>> stats = eval.trainAndTest(trainFiles, testFiles);
    AssertionEvaluation.printScore(stats, "target/models/neural/");
    double f1_ave = (stats.get("polarity").f1("-1") +
        stats.get("uncertainty").f1("1") +
        stats.get("generic").f1("true") +
        stats.get("conditional").f1("true") +
        stats.get("historyOf").f1("1")) / 5.0;
    
    System.out.println("Polarity: " + stats.get("polarity").f1("-1"));
    System.out.println("Uncertainty: " + stats.get("uncertainty").f1("1"));
    System.out.println("Generic: " + stats.get("generic").f1("true"));
    System.out.println("Conditional: " + stats.get("conditional").f1("true"));
    System.out.println("HistoryOf: " + stats.get("historyOf").f1("1"));
    System.out.println("Macro-f: " + f1_ave);
    
//    for(Map.Entry<String, AnnotationStatisticsCompact<String>> stat : stats.entrySet()){
//      System.out.println(stat.getKey());
//      System.out.println(stat.getValue().f1());
//    }
  }

  public static final Logger logger = UIMAFramework.getLogger(NeuralAssertionEvaluation.class);
  public boolean skipWrite = false;
  public boolean skipTrain = false;
  
  @Override
  protected CollectionReader getCollectionReader(List<File> items)
      throws Exception {
    String[] paths = new String[items.size()];
    for (int i = 0; i < paths.length; ++i) {
      paths[i] = items.get(i).getPath();
    }
    return CollectionReaderFactory.createReader(
        XMIReader.class,
        TypeSystemDescriptionFactory.createTypeSystemDescriptionFromPath(),
        XMIReader.PARAM_FILES,
        paths);
  }

  @Override
  protected void train(CollectionReader collectionReader, File directory)
      throws Exception {
    if(this.skipTrain) return;
    
    if(!this.skipWrite){
      AggregateBuilder builder = new AggregateBuilder();

      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralMultitaskAssertionStatusAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          true,
          DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
          KerasStringOutcomeDataWriter.class,
          DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
          directory,
          ScriptStringOutcomeDataWriter.PARAM_SCRIPT_DIRECTORY,
          "scripts/keras/"
          ) );

      // run the pipeline and write out the data
      SimplePipeline.runPipeline(collectionReader,  builder.createAggregateDescription());
    }
    
    // call the classifier builder to build a classifier and then package it into a jar
    JarClassifierBuilder.trainAndPackage(directory);
  }

  @Override
  protected Map<String, AnnotationStatisticsCompact<String>> test(
      CollectionReader collectionReader, File directory) throws Exception {
    Map<String,AnnotationStatisticsCompact<String>> stats = new HashMap<>();
    AnnotationStatisticsCompact<String> polarityStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> conditionalStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> uncertaintyStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> subjectStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> genericStats = new AnnotationStatisticsCompact<>();
    AnnotationStatisticsCompact<String> historyStats = new AnnotationStatisticsCompact<>();

    AggregateBuilder builder = new AggregateBuilder();

    // copy annotations to a gold view so we can evaluate 
    AnalysisEngineDescription goldCopierIdentifiedAnnotsAnnotator = AnalysisEngineFactory.createEngineDescription(ReferenceIdentifiedAnnotationsSystemToGoldCopier.class);
    builder.add(goldCopierIdentifiedAnnotsAnnotator);
    
    // remove gold from default view so we don't accidentally peak
    AnalysisEngineDescription assertionAttributeClearerAnnotator = AnalysisEngineFactory.createEngineDescription(ReferenceAnnotationsSystemAssertionClearer.class);
    builder.add(assertionAttributeClearerAnnotator);
    
    builder.add(AnalysisEngineFactory.createEngineDescription(NeuralMultitaskAssertionStatusAnalysisEngine.class,
        CleartkAnnotator.PARAM_IS_TRAINING,
        false,
        GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
        new File(directory, "model.jar").getPath()));

    JCasIterator casIter = new JCasIterator(collectionReader, builder.createAggregate());
    for ( ; casIter.hasNext();) {
      JCas jCas = casIter.next();
      JCas goldView;
      try {
        goldView = jCas.getView(AssertionEvaluation.GOLD_VIEW_NAME);
      } catch (CASException e) {
        logger.log(Level.INFO, "jCas.getViewName() = " + jCas.getViewName());
        throw new AnalysisEngineProcessException(e);
      }

      Collection<IdentifiedAnnotation> goldEntitiesAndEvents = getEntitiesAndEvents(goldView);
      Collection<IdentifiedAnnotation> systemEntitiesAndEvents = getEntitiesAndEvents(jCas);
      
      addStats(polarityStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "polarity");
      addStats(conditionalStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "conditional");
      addStats(uncertaintyStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "uncertainty");
      addStats(subjectStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "subject");
      addStats(genericStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "generic");
      addStats(historyStats, goldEntitiesAndEvents, systemEntitiesAndEvents, "historyOf");
    }
    casIter.collectionProcessComplete();
    
    stats.put("polarity",  polarityStats);
    stats.put("conditional",  conditionalStats);
    stats.put("uncertainty",  uncertaintyStats);
    stats.put("subject", subjectStats);
    stats.put("generic", genericStats);
    stats.put("historyOf", historyStats);

    return stats;
  }

  private static void addStats(AnnotationStatisticsCompact<String> stats, Collection<IdentifiedAnnotation> goldSpans, Collection<IdentifiedAnnotation> systemSpans, String key){
    stats.add(goldSpans, systemSpans,
        AnnotationStatisticsCompact.<IdentifiedAnnotation>annotationToSpan(),
        AnnotationStatisticsCompact.<IdentifiedAnnotation>annotationToFeatureValue(key));
  }
  private static Collection<IdentifiedAnnotation> getEntitiesAndEvents(JCas jCas){
    Collection<IdentifiedAnnotation> entitiesAndEvents = new ArrayList<>();
    Collection<EntityMention> entities = JCasUtil.select(jCas, EntityMention.class);
    entitiesAndEvents.addAll(entities);
    Collection<EventMention> events = JCasUtil.select(jCas, EventMention.class);
    entitiesAndEvents.addAll(events);
    return entitiesAndEvents;
  }
}
