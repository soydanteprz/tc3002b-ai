import java.io.*;
import java.util.*;
public class check1  {

//    public static void find(int n,int arr[])
//    {
//
//    }
//    2,1,3,4
    public static void main(String[] args) throws IOException{

        Reader sc=new Reader();
//        PrintWriter out = new PrintWriter(System.out);

        int t=sc.nextInt();
        while(t-->0)
        {
            int n = sc.nextInt();
            int p[]=new int[n];
            System.out.println("? 1");
            System.out.flush();
            int t1 = sc.nextInt();

            Set<Integer> hs = new HashSet<>();
            int i=1;
            int prev=1;
            int cnt=1;
            while(i<n && cnt<2*n)
            { 
                if(p[prev-1]!=0)
                {
                    i++;
                    if(i>=n || cnt>=2*n) break;
                    System.out.println("? "+i);
                    System.out.flush();
                    cnt++;

                    int t2 = sc.nextInt();
                    prev=t2;
                    continue;
                }
                String s1 = "? "+i;
                System.out.println(s1);
                System.out.flush();
                int tt = sc.nextInt();
                
                cnt++;                
                p[prev-1]=tt;
                prev=tt;
            }
            long sum=0;
            int ind=-1;
            for(i=0;i<n;i++)
            {
                sum+=p[i];
                if(p[i]==0) ind=i;
            }
            long diff = n*(n+1)-sum*2;
            if(diff>0)
            {
                p[ind]=(int)(diff/2);
            }
            System.out.print("!");
            for(i=0;i<n;i++)
            {
                System.out.print(" "+p[i]);
            }

//            out.println(Arrays.toString(p));
            System.out.println();
            System.out.flush();
        }
        System.out.flush();
    }

    

    //    (a*a_inverse) = 1(mod m)
//    finding a_inverse
    

    //    (x^power)mod(m)
    
    

    

    //    prime no smaller than or equal to n
    
    
    //       Arrays.sort(arr, new Comparator<pair>() {
//        //@Override
//        public int compare(pair o1, pair o2) {
//            long l1=o1.a-o2.a;
//            if(l1<0L) return -1;
//            if(l1==0) return 0;
//            return 1;
//        }
//    });
    

}
