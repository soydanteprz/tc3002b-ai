import javax.print.attribute.standard.MediaSize;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Stream;

public class Program {

    static BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
    static BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out));

    public static void main(String[] args) throws NumberFormatException, IOException {

        int cases = Integer.parseInt(reader.readLine());
        int mod = 998244353;

        while(cases-- > 0) {

            String[] firstLine = reader.readLine().split(" ");
            int n = Integer.parseInt(firstLine[0]);

            String a = reader.readLine();
            String b = reader.readLine();

            int ans = Integer.MAX_VALUE;

            int oneZeroPos = -1;
            int oneOnePos = -1;
            int oneZero = 0;
            int zeroOne = 0;

            for(int i=0;i<n;i++) {
                if(a.charAt(i)=='0' && b.charAt(i)=='1') {
                    zeroOne++;
                }
                if(a.charAt(i)=='1' && b.charAt(i)=='0') {
                    oneZero++;
                }
                if(a.charAt(i)=='1' && b.charAt(i)=='0' && oneZeroPos == -1) {
                    oneZeroPos = i;
                }
                if(a.charAt(i)=='1' && b.charAt(i)=='1' && oneOnePos == -1) {
                    oneOnePos = i;
                }
            }

            if(oneZero==0&&zeroOne==0) {
                printNumber(0);
                continue;
            }

            if(oneZero==zeroOne) {
                ans = oneZero+zeroOne;
            }

            int a1 = func(a, b, oneZeroPos);
            int b1 = func(a, b, oneOnePos);

            if(a1 != Integer.MAX_VALUE) {
                ans = Math.min(ans, 1+a1);
            }

            if(b1 != Integer.MAX_VALUE) {
                ans = Math.min(ans, 1+b1);
            }

            if(ans == Integer.MAX_VALUE) {
                printNumber(-1);
            } else {
                printNumber(ans);
            }

        }

        out.flush();

    }


    public static int func(String str, String comparing, int index) {

        if(index==-1) return Integer.MAX_VALUE;

        char[] ch = str.toCharArray();
        char[] comp = comparing.toCharArray();

        for(int i=0;i<ch.length;i++) {
            if(i!=index) {
                if(ch[i]=='1') ch[i] = '0';
                else ch[i] = '1';
            }
        }

        int oneZero = 0;
        int zeroOne = 0;

        for(int i=0;i<str.length();i++) {
            if(ch[i] == '1' && comp[i] == '0') {
                oneZero++;
            } else if(ch[i] == '0' && comp[i] == '1') {
                zeroOne++;
            }
        }

        if(oneZero == zeroOne) {
            return oneZero+zeroOne;
        }

        return Integer.MAX_VALUE;

    }

     

    

    

    

    

    

    

    public static void printNumber(long num) throws IOException {
        out.append(num + "\n");
    }

    

    

    

    

    

    

    

    

    


}
