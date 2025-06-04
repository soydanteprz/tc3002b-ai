import java.util.*;
import java.lang.*;
public class Codeforces {
    static Scanner sr=new Scanner(System.in);
    public static void main(String[] args) throws java.lang.Exception {
        StringBuilder ans = new StringBuilder("");
        int T = sr.nextInt();
        while (T-- > 0) {
            int n=sr.nextInt();
            int m=sr.nextInt();
            int x=sr.nextInt();
            TreeMap<Integer,ArrayList<Integer>>h=new TreeMap<>();
            for(int i=0;i<n;i++)
            {
                int a=sr.nextInt();
                if(!h.containsKey(a))
                    h.put(a,new ArrayList<>());
                h.get(a).add(i);
            }
            ans.append("YES");
            ans.append('\n');
            int an[]=new int[n];
            int q=0;
            for(int z:h.keySet())
            {
                for(int i=0;i<h.get(z).size();i++)
                {
                    an[h.get(z).get(i)]=(q++)%m+1;
                }
            }
            for(int i=0;i<n;i++)
                ans.append(an[i]+" ");
            ans.append('\n');
        }
        System.out.println(ans);
    }
}
