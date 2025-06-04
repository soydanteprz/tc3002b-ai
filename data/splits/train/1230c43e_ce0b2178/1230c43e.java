import java.io.*;
import java.math.BigInteger;
import java.util.*;

public class Main {

    static boolean[] primecheck = new boolean[1000002];

    public static void main(String[] args) throws IOException {
        OutputStream outputStream = System.out;
        FastReader in = new FastReader();
        PrintWriter out = new PrintWriter(outputStream);
        PROBLEM solver = new PROBLEM();
        int t = 1;
        t = in.nextInt();
        for (int i = 0; i < t; i++) {
            solver.solve(in, out);
        }
        out.close();
    }

    static class PROBLEM {

        public void solve(FastReader in, PrintWriter out) {

            int n = in.nextInt();
            int[] a = in.readArray(n);
            char[] c = in.nextLine().toCharArray();
            int cur = 1;

            ArrayList<Pair> p = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                p.add(new Pair(c[i], a[i]));
            }

            Collections.sort(p);
//            for (int i = 0; i < n; i++) {
//                out.println(p.get(i).x + " " + p.get(i).y);
//            }

            for (int i = 0; i < n; i++) {
                if(p.get(i).x == 'B' && p.get(i).y < i+1){
                    out.println("NO");
                    return;
                }
                if(p.get(i).x == 'R' && p.get(i).y > i+1){
                    out.println("NO");
                    return;
                }
            }

            out.println("YES");

//            int n = in.nextInt(), m = in.nextInt();
//            char[] s = in.nextLine().toCharArray();
//
//            int rl = 0, ud = 0, r = 0 , l = 0, rlf = 0, udf = 0;
//            int lmax = 0, rmax = 0, umax = 0, dmax = 0;
//
//            for (int i = 0; i < s.length; i++) {
//                if(s[i] == 'L'){
//                    if(rlf == 0) rlf = -1;
//                    rl--;
//                    l--;
//                    if(rl == 0) l = 0;
//                    if(rl < 0 && lmax+rmax <= m) lmax = Math.max(Math.abs(l), lmax);
//                }
//                if(s[i] == 'R'){
//                    if(rlf == 0) rlf = 1;
//                    rl++;
//                    r++;
//
//                    if(rl > 0 && lmax+rmax <= m) rmax = Math.max(r, rmax);
//                }
//                if(s[i] == 'U'){
//                    if(udf == 0) udf = 1;
//                    ud++;
//                    r = Math.max(Math.abs(ud), r);
//                    if(ud > 0 && umax+dmax <= n) umax = Math.max(Math.abs(ud), umax);
//                }
//                if(s[i] == 'D'){
//                    if(udf == 0) udf = -1;
//                    ud--;
//                    r = Math.max(Math.abs(ud), r);
//                    if(ud < 0 && umax+dmax <= n) dmax = Math.max(Math.abs(ud), dmax);
//                }
//            }
//
//            int ansc = 0, ansr = 0;
//
//            out.println(rlf + "  lmx = " + lmax + " rmax" + rmax);
//
//            if(rlf == 1) ansc = m-rmax;
//            else if(rlf == -1) ansc = 1+lmax;
//            else ansc = 1;
//
//            if(udf == 1) ansr = 1+umax;
//            else if(udf == -1) ansr = m-dmax;
//            else ansr = 1;
//
//            out.println(ansr + " " + ansc);

        }
    }

    

    

    

    

    

    

    

    

    


    static class Pair implements Comparable<Pair>{

        char x;
        int y;

        Pair(char x, int y){
            this.x = x;
            this.y = y;
        }

        
    }

    

//    public static class Pair<U extends Comparable<U>, V extends Comparable<V>> implements Comparable<Pair<U, V>> {
//        public U x;
//        public V y;
//
//        public Pair(U x, V y) {
//            this.x = x;
//            this.y = y;
//        }
//
//        public int hashCode() {
//            return (x == null ? 0 : x.hashCode() * 31) + (y == null ? 0 : y.hashCode());
//        }
//
//        public boolean equals(Object o) {
//            if (this == o)
//                return true;
//            if (o == null || getClass() != o.getClass())
//                return false;
//            Pair<U, V> p = (Pair<U, V>) o;
//            return (x == null ? p.x == null : x.equals(p.x)) && (y == null ? p.y == null : y.equals(p.y));
//        }
//
//        public int compareTo(Pair<U, V> b) {
//            int cmpU = x.compareTo(b.x);
//            return cmpU != 0 ? cmpU : y.compareTo(b.y);
//        }
//
//        public int compareToY(Pair<U, V> b) {
//            int cmpU = y.compareTo(b.y);
//            return cmpU != 0 ? cmpU : x.compareTo(b.x);
//        }
//
//        public String toString() {
//            return String.format("(%s, %s)", x.toString(), y.toString());
//        }
//
//    }

    

    

    

    
}
