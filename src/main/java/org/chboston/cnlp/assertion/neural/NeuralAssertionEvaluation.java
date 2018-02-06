package org.chboston.cnlp.assertion.neural;

import org.apache.commons.io.FileUtils;
import org.apache.ctakes.assertion.eval.AnnotationStatisticsCompact;
import org.apache.ctakes.assertion.eval.AssertionEvaluation;
import org.apache.ctakes.assertion.eval.AssertionEvaluation.ReferenceAnnotationsSystemAssertionClearer;
import org.apache.ctakes.assertion.eval.AssertionEvaluation.ReferenceIdentifiedAnnotationsSystemToGoldCopier;
import org.apache.ctakes.assertion.eval.XMIReader;
import org.apache.ctakes.neural.ScriptStringFeatureDataWriter;
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
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.util.*;

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
    
    @Option(
        name = "--write-only",
        usage = "Skip training and testing and just write the data with the data writer (for running parameter sweeps on the server)",
        required = false)
    public boolean writeOnly=false;
    
    @Option(
        name = "--baseline",
        usage = "Run baseline individual systems rather than multi-task system",
        required = false)
    public boolean baseline=false;
    
    @Option(
        name = "--data-dir",
        usage = "Directory where training data should be written",
        required=false)
    public String dataDir = "target/models";

    @Option(
        name = "--attributes",
        usage = "Colon-separated string containing attributes to detect (default is all)",
        required=false)
    public String rawAttributes = "polarity:uncertainty:conditional:generic:historyOf:subject";
  }
  
  public NeuralAssertionEvaluation(File baseDirectory) {
    super(baseDirectory);
  }

  public static void main(String[] args) throws Exception {
    Options options = new Options();
    CmdLineParser parser = new CmdLineParser(options);
    parser.parseArgument(args);

    File neuralDir = new File(options.dataDir);
    File configDir = new File(neuralDir, options.baseline ? "singletask" : "multitask");
    NeuralAssertionEvaluation eval = new NeuralAssertionEvaluation(configDir);

    eval.attributes = new HashSet(Arrays.asList(options.rawAttributes.split(":")));

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

    eval.baseline = options.baseline;
    eval.writeOnly = options.writeOnly;
    
    List<File> testFiles = new ArrayList<>();
    testFiles = Arrays.asList(options.testDirectory.listFiles());

    // if this is a new run make sure the models directory is empty:
    if(!(options.skipTrain || options.skipWrite)){
      FileUtils.deleteDirectory(new File(configDir, "train_and_test"));
    }
    
    Map<String, AnnotationStatisticsCompact<String>> stats = eval.trainAndTest(trainFiles, testFiles);
    if(options.writeOnly){
      System.out.println("No evaluation performed due to writeOnly flag");
      return;
    }
    
    AssertionEvaluation.printScore(stats, configDir.getAbsolutePath());
//    double f1_ave = (stats.get("polarity").f1("-1") +
//        stats.get("uncertainty").f1("1") +
//        stats.get("generic").f1("true") +
//        stats.get("conditional").f1("true") +
//        stats.get("historyOf").f1("1")) / 5.0;

    if(eval.attributes.contains("polarity")) System.out.println(String.format("Polarity: %.3f",  stats.get("polarity").f1("-1")));
    if(eval.attributes.contains("uncertainty")) System.out.println(String.format("Uncertainty: %.3f", stats.get("uncertainty").f1("1")));
    if(eval.attributes.contains("generic")) System.out.println(String.format("Generic: %.3f", stats.get("generic").f1("true")));
    if(eval.attributes.contains("conditional")) System.out.println(String.format("Conditional: %.3f", stats.get("conditional").f1("true")));
    if(eval.attributes.contains("historyOf")) System.out.println(String.format("HistoryOf: %.3f", stats.get("historyOf").f1("1")));
//    System.out.println(String.format("Macro-f: %.3f\n " , f1_ave));
    
//    for(Map.Entry<String, AnnotationStatisticsCompact<String>> stat : stats.entrySet()){
//      System.out.println(stat.getKey());
//      System.out.println(stat.getValue().f1());
//    }
  }

  public static final Logger logger = UIMAFramework.getLogger(NeuralAssertionEvaluation.class);
  public boolean skipWrite = false;
  public boolean skipTrain = false;
  public boolean baseline = false;
  public boolean writeOnly = false;
  public Set<String> attributes = null;
//  private String[] atts= new String[] {"polarity", "uncertainty", "conditional", "generic", "historyOf", "subject"};
//  private String[] i2b2Atts= new String[] {"polarity", "uncertainty", "conditional", "subject"};
//  private String[] atts = i2b2Atts;

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
    if (this.skipTrain) return;

    if (!this.skipWrite) {
      AggregateBuilder builder = new AggregateBuilder();

      if (baseline) {
        if (this.attributes.contains("polarity")) {
          builder.add(AnalysisEngineFactory.createEngineDescription(NeuralPolarityAnalysisEngine.class,
                  CleartkAnnotator.PARAM_IS_TRAINING,
                  true,
                  DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
                  ScriptStringFeatureDataWriter.class,
                  DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
                  new File(directory, "polarity"),
                  ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
                  "scripts/keras/singletask/polarity"
          ));
        }
        if(this.attributes.contains("uncertainty")) {
          builder.add(AnalysisEngineFactory.createEngineDescription(NeuralUncertaintyAnalysisEngine.class,
                  CleartkAnnotator.PARAM_IS_TRAINING,
                  true,
                  DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
                  ScriptStringFeatureDataWriter.class,
                  DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
                  new File(directory, "uncertainty"),
                  ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
                  "scripts/keras/singletask/uncertainty"
          ));
        }
        if(this.attributes.contains("conditional")) {
          builder.add(AnalysisEngineFactory.createEngineDescription(NeuralConditionalAnalysisEngine.class,
                  CleartkAnnotator.PARAM_IS_TRAINING,
                  true,
                  DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
                  ScriptStringFeatureDataWriter.class,
                  DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
                  new File(directory, "conditional"),
                  ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
                  "scripts/keras/singletask/conditional"
          ));
        }
        if(this.attributes.contains("generic")) {
          builder.add(AnalysisEngineFactory.createEngineDescription(NeuralGenericAnalysisEngine.class,
                  CleartkAnnotator.PARAM_IS_TRAINING,
                  true,
                  DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
                  ScriptStringFeatureDataWriter.class,
                  DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
                  new File(directory, "generic"),
                  ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
                  "scripts/keras/singletask/generic"
          ));
        }
        if(this.attributes.contains("historyOf")) {
          builder.add(AnalysisEngineFactory.createEngineDescription(NeuralHistoryOfAnalysisEngine.class,
                  CleartkAnnotator.PARAM_IS_TRAINING,
                  true,
                  DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
                  ScriptStringFeatureDataWriter.class,
                  DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
                  new File(directory, "historyOf"),
                  ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
                  "scripts/keras/singletask/historyOf"
          ));
        }
        if(this.attributes.contains("subject")) {
          builder.add(AnalysisEngineFactory.createEngineDescription(NeuralSubjectAnalysisEngine.class,
                  CleartkAnnotator.PARAM_IS_TRAINING,
                  true,
                  DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
                  ScriptStringFeatureDataWriter.class,
                  DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
                  new File(directory, "subject"),
                  ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
                  "scripts/keras/singletask/subject"
          ));
        }
      } else {
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralMultitaskAssertionStatusAnalysisEngine.class,
                CleartkAnnotator.PARAM_IS_TRAINING,
                true,
                DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
                ScriptStringFeatureDataWriter.class,
                DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
                directory,
                ScriptStringFeatureDataWriter.PARAM_SCRIPT_DIR,
                "scripts/keras/multitask/"
        ));
      }
      // run the pipeline and write out the data
      SimplePipeline.runPipeline(collectionReader, builder.createAggregateDescription());
    }
    if (this.writeOnly) return;

    // call the classifier builder to build a classifier and then package it into a jar
    if (this.baseline) {
      for (String attribute : attributes) {
        JarClassifierBuilder.trainAndPackage(new File(directory, attribute));
      }
    } else {
      JarClassifierBuilder.trainAndPackage(directory);
    }
  }

  @Override
  protected Map<String, AnnotationStatisticsCompact<String>> test(
      CollectionReader collectionReader, File directory) throws Exception {
    Map<String,AnnotationStatisticsCompact<String>> stats = new HashMap<>();
    if(this.writeOnly) return stats;
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
    
    if(this.baseline){
      if(this.attributes.contains("polarity")) {
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralPolarityAnalysisEngine.class,
                CleartkAnnotator.PARAM_IS_TRAINING,
                false,
                GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
                new File(new File(directory, "polarity"), "model.jar").getPath()));
      }
      if(this.attributes.contains("uncertainty")) {
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralUncertaintyAnalysisEngine.class,
                CleartkAnnotator.PARAM_IS_TRAINING,
                false,
                GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
                new File(new File(directory, "uncertainty"), "model.jar").getPath()));
      }
      if(this.attributes.contains("conditional")) {
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralConditionalAnalysisEngine.class,
                CleartkAnnotator.PARAM_IS_TRAINING,
                false,
                GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
                new File(new File(directory, "conditional"), "model.jar").getPath()));
      }
      if(this.attributes.contains("generic")) {
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralGenericAnalysisEngine.class,
                CleartkAnnotator.PARAM_IS_TRAINING,
                false,
                GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
                new File(new File(directory, "generic"), "model.jar").getPath()));
      }
      if(this.attributes.contains("historyOf")) {
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralHistoryOfAnalysisEngine.class,
                CleartkAnnotator.PARAM_IS_TRAINING,
                false,
                GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
                new File(new File(directory, "historyOf"), "model.jar").getPath()));
      }
      if(this.attributes.contains("subject")) {
        builder.add(AnalysisEngineFactory.createEngineDescription(NeuralSubjectAnalysisEngine.class,
                CleartkAnnotator.PARAM_IS_TRAINING,
                false,
                GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
                new File(new File(directory, "subject"), "model.jar").getPath()));
      }
    }else{
      builder.add(AnalysisEngineFactory.createEngineDescription(NeuralMultitaskAssertionStatusAnalysisEngine.class,
          CleartkAnnotator.PARAM_IS_TRAINING,
          false,
          GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
          new File(directory, "model.jar").getPath()));
    }
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

    if(this.attributes.contains("polarity")) stats.put("polarity",  polarityStats);
    if(this.attributes.contains("conditional")) stats.put("conditional",  conditionalStats);
    if(this.attributes.contains("uncertainty")) stats.put("uncertainty",  uncertaintyStats);
    if(this.attributes.contains("subject")) stats.put("subject", subjectStats);
    if(this.attributes.contains("generic")) stats.put("generic", genericStats);
    if(this.attributes.contains("historyOf")) stats.put("historyOf", historyStats);

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
