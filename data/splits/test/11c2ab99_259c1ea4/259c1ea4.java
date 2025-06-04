import java.util.Arrays;
import java.util.Scanner;

public class E {
    public static void main(String[] args) {
        Scanner scn = new Scanner(System.in);
        int test = scn.nextInt();
        while (test-- > 0) {
            int nrLands = scn.nextInt();
            int[] toRightLands = new int[nrLands];
            Arrays.fill(toRightLands, (int) (1e9 + 1e8));
            int[] toLeftLands = new int[nrLands];
            Arrays.fill(toLeftLands, (int) (1e9 + 1e8));
            int nrAcs = scn.nextInt();

            AC[] acs = new AC[nrAcs];
            for (int i = 0; i < nrAcs; i++) {
                acs[i] = new AC();
                acs[i].myPos = scn.nextInt() - 1;
            }
            for (int i = 0; i < nrAcs; i++) {
                acs[i].myTemp = scn.nextInt();
            }
            Arrays.sort(acs);
            int nextAc = 1;
            AC lastChosen = acs[0];

            for (int i = lastChosen.myPos; i < nrLands; i++) {

                if (nextAc < acs.length && i == acs[nextAc].myPos) {
                    if (acs[nextAc].myTemp < lastChosen.myTemp + i - lastChosen.myPos) {
                        lastChosen = acs[nextAc];
                    }
                    nextAc++;
                }

                toRightLands[i] = lastChosen.myTemp + i - lastChosen.myPos;

            }


            int nextAc1 = acs.length - 2;
            AC lastChosen1 = acs[acs.length - 1];

            for (int i = lastChosen1.myPos; i >= 0; i--) {
                if (nextAc1 >= 0 && i == acs[nextAc1].myPos) {
                    if (acs[nextAc1].myTemp < lastChosen1.myTemp - i + lastChosen1.myPos) {
                        lastChosen1 = acs[nextAc1];
                    }
                    nextAc1--;
                }

                toLeftLands[i] = lastChosen1.myTemp - i + lastChosen1.myPos;
            }
            for (int i = 0; i < nrLands; i++) {
                System.out.print(Integer.min(toLeftLands[i], toRightLands[i]) + " ");
            }
            System.out.println();


        }


    }


    static class AC implements Comparable<AC> {
        int myPos, myTemp;


        
    }
}
