import java.util.Scanner;
import java.util.TreeSet;

public class D {
    public static void main(String[] args) {
        Scanner scn = new Scanner(System.in);
        int test = scn.nextInt();
        for (int t = 0; t < test; t++) {
            int n = scn.nextInt();
            int[] b = new int[n];
            TreeSet<Integer> ts = new TreeSet<>();
            int lastAdded = -1;
            boolean ans = true;
            for (int i = 0; i < n; i++) {
                b[i] = scn.nextInt();
            }
            for (int val : b) {
                if (t == 4) {
                    int x = 2;
                }
                if (val > lastAdded) {
                    Integer between = ts.higher(lastAdded);
                    if (between != null && between < val) {
                        ans = false;
                        break;

                    }

                }
                if (val < lastAdded) {
                    Integer between = ts.lower(lastAdded);
                    if (between != null && between > val) {
                        ans = false;
                        break;
                    }
                }
                ts.add(val);
                lastAdded = val;
            }
            if (ans) System.out.println("YES");
            else System.out.println("NO");
        }

    }

    


}
