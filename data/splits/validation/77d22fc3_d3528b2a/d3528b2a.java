import java.io.*;
import java.util.*;

public class A {
    //--------------------------INPUT READER---------------------------------//

    //-----------------------------------------------------------------------//
    static int t = 0;
    //---------------------------PRINTER-------------------------------------//
    //-----------------------------------------------------------------------//

    //--------------------------VARIABLES------------------------------------//
    static FastReader sc = new FastReader();
    static OutputStream outputStream = System.out;
    static long ima = Integer.MAX_VALUE;
    //-----------------------------------------------------------------------//

    //--------------------------ADMIN_MODE-----------------------------------//
    private static void ADMIN_MODE() throws IOException {
        if (System.getProperty("ONLINE_JUDGE") == null) {
            sc = new FastReader(new FileInputStream("input.txt"));
        }
    }
    //-----------------------------------------------------------------------//
    static int tt = 0;
    //----------------------------START--------------------------------------//
    public static void main(String[] args)
            throws IOException {

        ADMIN_MODE();

        t = sc.ni();while(t-->0) {
            tt++;
            solve();
        }


    }

    static void solve() throws IOException {
        int n = sc.ni();
        String s1 = sc.ns();
        String s2 = sc.ns();
        char[] strr = s1.toCharArray();
        char[] strr2 = s2.toCharArray();
        HashSet<Integer> lia = new HashSet<>(), lib = new HashSet<>();
        for(int i = 0; i < n; i++) {
            if(strr[i]=='1') lia.add(i);
            else lib.add(i);
        }
        HashSet<Integer> liaa = new HashSet<>();
        for(int i = 0; i < n; i++) {
            if(strr2[i]=='1') liaa.add(i);
        }
        if(!(lia.size() == liaa.size() || lib.size()+1 == liaa.size())) {
            System.out.println(-1);
            return;
        }
        int ac = 0, bc = 0;
        for(int i: lia) {
            if(liaa.contains(i)) ac++;
        }
        bc = liaa.size()-ac;
        if(lia.size() == liaa.size() && lia.size() == ac) {
            System.out.println(0);
            return;
        }

        int nac = lia.size()-ac;
        int nbc = lib.size()-bc;

        long ansa = ac != 0 && (ac-1 == nbc)? Math.abs(ac-1+nbc): ima;
        long ansb = nac != 0 && (nac-1 == bc-1)? Math.abs(nac-1+bc): ima;


        if(ansa == ima && ansb == ima) {
            System.out.println(-1);
            return;
        }

        System.out.println(Math.min(ansa, ansb)+1);
    }

}
