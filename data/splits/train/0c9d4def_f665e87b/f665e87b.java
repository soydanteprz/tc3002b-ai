import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;
import java.util.TreeMap;

public class C_CF {

    public static void main(String[] args) {
        FastScanner57 fs = new FastScanner57();
        PrintWriter pw = new PrintWriter(System.out);
        int t = fs.ni();
        //int t = 1;
        for (int tc = 0; tc < t; tc++) {
            int n = fs.ni();
            String[] s = new String[n];
            for (int i = 0; i < n; i++) {
                s[i] = fs.next();
            }
            int res = c(0,s);
            for (int i = 1; i < 5; i++) {
                res = Math.max(res,c(i,s));
            }
            pw.println(res);
        }
        
        pw.close();

    }
    public static int c(int l, String[] s) {
        List<Integer> list = new ArrayList();
        for (int i = 0; i < s.length; i++) {
            String t = s[i];
            int ct = 0;
            for (int j = 0; j < t.length(); j++) {
                if (t.charAt(j)-'a'==l) {
                    ct++;
                } else {
                    ct--;
                }
            }
            list.add(ct);
        }
        Collections.sort(list);
        Collections.reverse(list);
        int sum = 0;
        for (int i = 0; i < s.length; i++) {
            sum += list.get(i);
            if (sum<=0) return i;
        }
        return list.size();
    }
    

    

    // 0 -> left was chosen
    // 1 -> right was chosen

    public static void sort(long[] a) {
        List<Long> list = new ArrayList();
        for (int i = 0; i < a.length; i++) {
            list.add(a[i]);
        }
        Collections.sort(list);
        for (int i = 0; i < a.length; i++) {
            a[i] = list.get(i);
        }
    }

    

}




