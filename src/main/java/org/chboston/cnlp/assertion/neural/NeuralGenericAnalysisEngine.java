package org.chboston.cnlp.assertion.neural;

import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;

public class NeuralGenericAnalysisEngine extends
    NeuralAssertionStatusAnalysisEngine {

  @Override
  public String getLabel(IdentifiedAnnotation ent) {
    return String.valueOf(ent.getGeneric());
  }

  @Override
  public void applyLabel(IdentifiedAnnotation ent, String label) {
    ent.setGeneric(Boolean.parseBoolean(label));
  }

}
