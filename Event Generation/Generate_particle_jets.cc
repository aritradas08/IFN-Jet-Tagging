#include <iostream>
#include "Pythia8/Pythia.h"
#include <fstream>
#include <vector>
#include "fastjet/ClusterSequence.hh"
#include "IFNPlugin.hh"
#include <iomanip>
#include "fastjet/PseudoJet.hh"
#include "FlavNeutraliser.hh"

using namespace Pythia8;
using namespace std;
using namespace fastjet;
using namespace fastjet::contrib;

// Helper function to remove square brackets from flavour description
string strip_brackets(const string& s) {
    if (s.size() >= 2 && s.front() == '[' && s.back() == ']') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

int main() {
    int nevents = 5000;

    Pythia pythia;

    // Set fixed seed for reproducibility
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = 2");

    // Proton-proton collisions
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13600");

    // Enable hard QCD processes
    pythia.readString("HardQCD:all = on");
    pythia.readString("PhaseSpace:pTHatMin = 5000");

    // Turn ON hadronization and parton-level evolution
    pythia.readString("PartonLevel:all = on");
    pythia.readString("HadronLevel:all = on");
    
    pythia.init();
    
    ofstream outfile("particle_ifn_jets.txt");
    outfile << "event_number\tpt\t\tE\t\trap\tphi\tflav\tATLAS_tag\n";

    JetDefinition base_jet_def(antikt_algorithm, 0.4);
    fastjet::contrib::FlavRecombiner flav_recombiner;
    base_jet_def.set_recombiner(&flav_recombiner);

    double alpha = 2.0;
    double omega = 3.0 - alpha;
    fastjet::contrib::FlavRecombiner::FlavSummation flav_summation = fastjet::contrib::FlavRecombiner::net;
    auto ifn_plugin = new fastjet::contrib::IFNPlugin(base_jet_def, alpha, omega, flav_summation);
    JetDefinition IFN_jet_def(ifn_plugin);
    IFN_jet_def.delete_plugin_when_unused();
    
    double ptmin = 20.0;
    
    for (int i = 0; i < nevents; i++) {
        if (!pythia.next()) continue;
        
        vector<PseudoJet> event;
        
        for (int j = 0; j < pythia.event.size(); j++) {
            if (pythia.event[j].isFinal()) {
                int id = pythia.event[j].id();
                double px = pythia.event[j].px();
                double py = pythia.event[j].py();
                double pz = pythia.event[j].pz();
                double E = pythia.event[j].e();
                
                PseudoJet p(px, py, pz, E);
                p.set_user_info(new FlavHistory(id));
                event.push_back(p);
            }
        }
        
        vector<PseudoJet> base_jets = base_jet_def(event);
        vector<PseudoJet> IFN_jets  = IFN_jet_def(event);
        
        assert(base_jets.size() == IFN_jets.size());
        
        for (unsigned int ijet = 0; ijet < base_jets.size(); ijet++) {
            const auto & base_jet = base_jets[ijet];
            const auto & IFN_jet  = IFN_jets [ijet];
            
            if (base_jet.pt() < ptmin || IFN_jet.pt() < ptmin) continue;

            auto constituents = sorted_by_E(IFN_jet.constituents());
            string atlas_tag = "[unknown]";
            if (!constituents.empty()) {
                atlas_tag = FlavHistory::current_flavour_of(constituents.front()).description();
            }

            // Write to file: event_number pt E rap phi flav atlas_tag
            outfile << i + 1 << "\t"
                    << fixed << setprecision(3)
                    << IFN_jet.pt() << "\t"
                    << IFN_jet.e()  << "\t"
                    << IFN_jet.rap() << "\t"
                    << IFN_jet.phi() << "\t"
                    << strip_brackets(FlavHistory::current_flavour_of(IFN_jet).description()) << "\t"
                    << strip_brackets(atlas_tag) << "\n";
        }
    }
    
    outfile.close();
    return 0;
}

