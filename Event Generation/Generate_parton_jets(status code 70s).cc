#include <iostream>
#include "Pythia8/Pythia.h"
#include <fstream>
#include <vector>
#include <iomanip>

#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"
#include "IFNPlugin.hh"
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

    ofstream outfile("ifn_from_status61to69_partons.txt");
    outfile << "event_number\tpt\t\tE\t\trap\tphi\tflav\tATLAS_tag\n";

    // Jet definition with flavour recombiner
    JetDefinition base_jet_def(antikt_algorithm, 0.4);
    fastjet::contrib::FlavRecombiner flav_recombiner;
    base_jet_def.set_recombiner(&flav_recombiner);

    // Set IFN plugin parameters
    double alpha = 2.0;
    double omega = 3.0 - alpha;
    fastjet::contrib::FlavRecombiner::FlavSummation flav_summation = fastjet::contrib::FlavRecombiner::net;

    auto ifn_plugin = new fastjet::contrib::IFNPlugin(base_jet_def, alpha, omega, flav_summation);
    JetDefinition IFN_jet_def(ifn_plugin);
    IFN_jet_def.delete_plugin_when_unused();

    double ptmin = 20.0;

    for (int i = 0; i < nevents; i++) {
        if (!pythia.next()) continue;

        vector<PseudoJet> selected_partons;

        for (int j = 0; j < pythia.event.size(); j++) {
            const Particle& parton = pythia.event[j];
            int status = abs(parton.status());
            int id = abs(parton.id());

            // Check if parton is in status 71–79 and is a quark or gluon
            if (status >= 71 && status <= 79 &&
                ((id >= 1 && id <= 6) || id == 21)) {

                bool has_qg_daughter = false;

                // Check if any daughter is a quark or gluon
                for (int d = parton.daughter1(); d <= parton.daughter2(); ++d) {
                    if (d < 0 || d >= pythia.event.size()) continue;
                    int daughter_id = abs(pythia.event[d].id());
                    if ((daughter_id >= 1 && daughter_id <= 6) || daughter_id == 21) {
                        has_qg_daughter = true;
                        break;
                    }
                }

                // Only accept terminal quarks or gluons
                if (!has_qg_daughter) {
                    double px = parton.px();
                    double py = parton.py();
                    double pz = parton.pz();
                    double E  = parton.e();
                    PseudoJet p(px, py, pz, E);
                    p.set_user_info(new FlavHistory(parton.id()));
                    selected_partons.push_back(p);
                }
            }
        }

        // Run IFN clustering
        vector<PseudoJet> IFN_jets = IFN_jet_def(selected_partons);

        for (unsigned int ijet = 0; ijet < IFN_jets.size(); ijet++) {
            const auto& jet = IFN_jets[ijet];
            if (jet.pt() < ptmin) continue;

            auto constituents = sorted_by_E(jet.constituents());
            string atlas_tag = "[unknown]";
            if (!constituents.empty()) {
                atlas_tag = FlavHistory::current_flavour_of(constituents.front()).description();
            }

            outfile << i + 1 << "\t"
                    << fixed << setprecision(3)
                    << jet.pt()  << "\t"
                    << jet.e()   << "\t"
                    << jet.rap() << "\t"
                    << jet.phi() << "\t"
                    << strip_brackets(FlavHistory::current_flavour_of(jet).description()) << "\t"
                    << strip_brackets(atlas_tag) << "\n";
        }
    }

    outfile.close();
    return 0;
}

