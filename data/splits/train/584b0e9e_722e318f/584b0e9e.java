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


public class D_Round_753_Div3 {
    public static int MOD = 1000000007;

    static int[][] dp;


    public static void main(String[] args) throws FileNotFoundException {
        // PrintWriter out = new PrintWriter(new FileOutputStream(new File(
        // "output.txt")));
        PrintWriter out = new PrintWriter(System.out);
        Scanner in = new Scanner();
        int T = in.nextInt();
        for (int z = 0; z < T; z++) {
            int n = in.nextInt();
            int[] data = new int[n];
            for (int i = 0; i < n; i++) {
                data[i] = in.nextInt();
            }
            String line = in.next();
            ArrayList<Integer> blue = new ArrayList<>();
            ArrayList<Integer> red = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (line.charAt(i) == 'B') {
                    blue.add(data[i]);
                } else {
                    red.add(data[i]);
                }
            }
            Collections.sort(blue);
            Collections.sort(red);
            int st = 1;
            boolean ok = true;
            for (int i : blue) {
                if (i < st) {
                    ok = false;
                    break;
                }
                st++;
            }
            if (ok) {
                for (int i : red) {
                    if (i > st) {
                        ok = false;
                        break;
                    }
                    st++;
                }
            }
            out.println(ok ? "Yes" : "No");
        }
        out.close();
    }

    


    


    

    

    

    

    

    

    

    

    
}
