import java.util.*;
import java.lang.Math;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import javax.management.Query;
import java.io.*;
import java.math.BigInteger;

public class Main {
    static int mod = 1000000007;

    /* ======================DSU===================== */
    

    /* ======================================================== */
    

    

    static FastReader f = new FastReader();
    static BufferedWriter w = new BufferedWriter(new OutputStreamWriter(System.out));

    

    

    

    

    

    

    

    

    

    

    /*
     * ===========Modular Operations==================
     */
    

    

    

    

    /*
     * ===============================================
     */
    

    

    

    /*
     * ===========Dynamic prog Recur Section===========
     */

    static int DP[][];
    static ArrayList<ArrayList<Integer>> g;
    static int count = 0;

    

    /*
     * ====================================Main=================================
     */
    public static void main(String args[]) throws Exception {
        // File file = new File("D:\\VS Code\\Java\\Output.txt");
        // FileWriter fw = new FileWriter("D:\\VS Code\\Java\\Output.txt");
        Random rand = new Random();
        int t = 1;
        t = f.nextInt();
        int tc = 1;
        while (t-- != 0) {
            int n = f.nextInt();
            int c[] = new int[n];
            long minOdd = 0, minEven = 0;
            long sumEven = 0, sumOdd = 0;
            for (int i = 0; i < n ; i++) {
                c[i] = f.nextInt();
                // if (i % 2 == 0) {
                //     minEven = (c[minEven] > c[i]) ? i : minEven;
                //     sumEven += c[i];
                // } else {
                //     minOdd = (minOdd > c[i]) ? i : minOdd;
                //     sumOdd += c[i];
                // }
            }
            minEven = c[0];
            minOdd = c[1];
            sumEven=c[0];
            sumOdd=c[1];
            long min=minEven*n + minOdd*n;//for k=2
            int even=1,odd=1;
            for (int k = 3; k <= n; k++) {
                if(k%2==1){
                    sumEven+=c[k-1];
                    minEven=Math.min(minEven, c[k-1]);
                    even++;
                }else{
                    sumOdd+=c[k-1];
                    minOdd=Math.min(minOdd, c[k-1]);
                    odd++;
                }
                min=Math.min(min, sumEven-minEven+minEven*(n-even+1) + sumOdd-minOdd+minOdd*(n-odd+1));
            }
            w.write(min+"\n");

        }
        w.flush();
    }
}
