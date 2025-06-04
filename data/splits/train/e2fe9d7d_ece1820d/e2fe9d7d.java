import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Calendar;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.SortedSet;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.TreeSet;

public class C_Edu_Round_120 {
    public static long MOD = 998244353;

    static int[] dp;


    public static void main(String[] args) throws FileNotFoundException {
        // PrintWriter out = new PrintWriter(new FileOutputStream(new File(
        // "output.txt")));
        PrintWriter out = new PrintWriter(System.out);
        Scanner in = new Scanner();
        int T = in.nextInt();
        //System.out.println(cal(1608737403, 1000000000) - 923456789987654321L);
        for (int z = 0; z < T; z++) {
            int n = in.nextInt();
            long k = in.nextLong();
            long[] data = new long[n];
            long total = 0;
            PriorityQueue<Integer> q = new PriorityQueue<>();
            for (int i = 0; i < n; i++) {
                q.add(in.nextInt());
            }
            for (int i = 0; i < n; i++) {
                data[i] = q.poll();
                total += data[i];
            }
            Arrays.sort(data);
            long result = Long.max(0, total - k);
            long cur = 0;
            int num = 0;
            for (int i = n - 1; i > 0 && total > k + cur; i--) {

                num++;
                cur += data[i] - data[0];
                long left = Long.max(0, total - cur - k);
                left /= (num + 1);
                if (total - cur - (left * (num + 1)) > k) {
                    left++;
                }
                //System.out.println(num + " " + left + " " + cur + " " + total);
                result = Long.min(result, left + num);
            }


            out.println(result);

        }

        out.close();
    }

    

    

    

    


    

    

    

    

    

    

    

    

    
}
